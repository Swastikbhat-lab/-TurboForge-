"""
TurboForge - Main Pipeline
Full end-to-end training, evaluation, and inference pipeline.

Usage:
    python main.py --mode train --epochs 50
    python main.py --mode infer
"""

import argparse
import numpy as np
import pandas as pd
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from data.scada_generator import generate_scada_data
from utils.preprocessing import SCADAPreprocessor
from models.turboforge_transformer import TurboForgeTransformer, TurboForgeTrainer
from models.temporal_gan import TemporalGAN
from models.llm_explainer import TurboForgeExplainer, TurbineStatus


CONFIG = {
    "n_turbines": 50,
    "seq_len": 36,
    "feature_dim": 9,
    "d_model": 128,
    "temporal_heads": 4,
    "spatial_heads": 8,
    "batch_size": 8,
    "epochs": 50,
    "lr": 1e-4,
    "gan_epochs": 100,
    "data_hours": 2160,
    "pos_weight": 10.0,     # Fix class imbalance — penalize missed failures 10x
}


def load_or_generate_data(csv_path: str = "scada_data.csv"):
    if os.path.exists(csv_path):
        print(f"[Data] Loading existing data from {csv_path}")
        return pd.read_csv(csv_path, parse_dates=["timestamp"])
    else:
        print(f"[Data] Generating synthetic SCADA data...")
        df = generate_scada_data(n_turbines=CONFIG["n_turbines"], n_hours=CONFIG["data_hours"])
        df.to_csv(csv_path, index=False)
        return df


def train_gan(df, preprocessor):
    print("\n" + "=" * 60)
    print("STEP 1: Training Temporal GAN")
    print("=" * 60)
    turbine_df = df[df["turbine_id"] == 1].sort_values("timestamp")
    X_scaled = preprocessor.scaler.fit_transform(
        turbine_df[["wind_speed_ms","rotor_rpm","power_output_kw","blade_pitch_deg",
                    "nacelle_temp_c","gearbox_temp_c","generator_temp_c",
                    "vibration_x","vibration_y"]].values
    )
    gan = TemporalGAN(seq_len=CONFIG["seq_len"], feature_dim=CONFIG["feature_dim"])
    gan.fit(X_scaled, epochs=CONFIG["gan_epochs"], batch_size=64)
    gan.save("temporal_gan.pt")
    print(f"[GAN] Generated {len(gan.generate(500))} synthetic sequences")
    return gan


def train_transformer(df, preprocessor):
    print("\n" + "=" * 60)
    print("STEP 2: Training TurboForge Transformer")
    print("=" * 60)

    X, y = preprocessor.create_sequences_per_turbine(df)
    train_loader, val_loader, test_loader = preprocessor.get_dataloaders(X, y, batch_size=CONFIG["batch_size"])

    model = TurboForgeTransformer(
        n_turbines=CONFIG["n_turbines"],
        feature_dim=CONFIG["feature_dim"],
        d_model=CONFIG["d_model"],
        temporal_heads=CONFIG["temporal_heads"],
        spatial_heads=CONFIG["spatial_heads"],
    )

    trainer = TurboForgeTrainer(model, lr=CONFIG["lr"], pos_weight=CONFIG["pos_weight"])
    trainer.fit(train_loader, val_loader, epochs=CONFIG["epochs"])

    print("\n[Final Test Evaluation]")
    test_metrics = trainer.evaluate(test_loader)
    print(f"  Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Test ROC-AUC:   {test_metrics['roc_auc']:.4f}")

    return model


def run_inference(model, df, preprocessor):
    print("\n" + "=" * 60)
    print("STEP 3: Inference + LLM Explanations")
    print("=" * 60)

    model.eval()
    device = next(model.parameters()).device
    X, y = preprocessor.create_sequences_per_turbine(df.tail(CONFIG["n_turbines"] * 200))
    X_tensor = torch.FloatTensor(X[-1:]).to(device)

    with torch.no_grad():
        failure_probs, coord_score = model(X_tensor)

    failure_probs = failure_probs.squeeze(0).cpu().numpy()
    print(f"\n[Inference] Fleet Coordination Score: {coord_score.item():.3f}")
    print(f"[Inference] Mean Failure Probability:  {failure_probs.mean():.3f}")
    print(f"[Inference] Turbines at risk (>50%):   {(failure_probs > 0.5).sum()}")

    explainer = TurboForgeExplainer()
    turbine_statuses = []

    for tid in range(CONFIG["n_turbines"]):
        tdf = df[df["turbine_id"] == tid + 1].tail(1)
        if tdf.empty:
            continue
        sensor_vals = {
            col: float(tdf[col].values[0])
            for col in ["wind_speed_ms","rotor_rpm","power_output_kw","blade_pitch_deg",
                        "nacelle_temp_c","gearbox_temp_c","generator_temp_c","vibration_x","vibration_y"]
        }
        risk_features = sorted(sensor_vals.keys(), key=lambda k: abs(sensor_vals[k]), reverse=True)[:3]
        turbine_statuses.append(TurbineStatus(
            turbine_id=tid + 1,
            failure_probability=float(failure_probs[tid]),
            sensor_values=sensor_vals,
            top_risk_features=risk_features,
        ))

    print("\n[LLM] Generating fleet summary...")
    print(explainer.fleet_summary(turbine_statuses))

    top = max(turbine_statuses, key=lambda s: s.failure_probability)
    print(f"\n[LLM] Turbine {top.turbine_id} detail (Risk: {top.failure_probability:.1%})")
    print(explainer.explain_failure(top))

    return turbine_statuses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer", "full"], default="full")
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--data_csv", type=str, default="scada_data.csv")
    parser.add_argument("--skip_gan", action="store_true")
    parser.add_argument("--n_turbines", type=int, default=CONFIG["n_turbines"])
    parser.add_argument("--data_hours", type=int, default=CONFIG["data_hours"])
    args = parser.parse_args()

    CONFIG["epochs"] = args.epochs
    CONFIG["n_turbines"] = args.n_turbines
    CONFIG["data_hours"] = args.data_hours

    print("=" * 60)
    print("  TurboForge - Wind Farm Digital Twin")
    print("  Generative AI + Transformers for Failure Prediction")
    print("=" * 60)

    df = load_or_generate_data(csv_path=args.data_csv)
    preprocessor = SCADAPreprocessor(seq_len=CONFIG["seq_len"])

    if args.mode in ["train", "full"]:
        if not args.skip_gan:
            train_gan(df, preprocessor)
        model = train_transformer(df, preprocessor)
        if args.mode == "full":
            run_inference(model, df, preprocessor)

    elif args.mode == "infer":
        model = TurboForgeTransformer(
            n_turbines=CONFIG["n_turbines"], feature_dim=CONFIG["feature_dim"],
            d_model=CONFIG["d_model"], temporal_heads=CONFIG["temporal_heads"],
            spatial_heads=CONFIG["spatial_heads"],
        )
        model.load_state_dict(torch.load("best_turboforge.pt", map_location="cpu"))
        run_inference(model, df, preprocessor)

    print("\n✅ TurboForge pipeline complete.")


if __name__ == "__main__":
    main()
