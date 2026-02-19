import sys, os
sys.path.insert(0, os.getcwd())
from data.scada_generator import generate_scada_data
df = generate_scada_data(n_turbines=50, n_hours=2160)
df.to_csv('scada_data.csv', index=False)
print(f'[OK] Generated {len(df):,} records saved to scada_data.csv')
