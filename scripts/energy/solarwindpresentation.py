import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class CONFIG:

    NAMES_DTYPES = {
        "Source" : str,
        "Production" : np.float32
    }

production_fr_df = pd.read_csv('scripts/energy/intermittent-renewables-production-france.csv', sep=',',
    index_col="Date and Hour",
    parse_dates=["Date and Hour", "Date"],
    infer_datetime_format=True,
    dtype=CONFIG.NAMES_DTYPES
    )

plt.figure(figsize=(10, 6))

# Tracer la production éolienne en bleu et solaire en rouge
production_fr_df[production_fr_df['Source'] == 'Wind']['Production'].plot(label='Wind Production', color='blue')
production_fr_df[production_fr_df['Source'] == 'Solar']['Production'].plot(label='Solar Production', color='red')

plt.xlabel('Date and Hour')
plt.ylabel('Production (MWh)')
plt.title('Production en fonction de la Date et de l\'Heure')
plt.legend()
plt.savefig('static/images/energy/solar_wind_presentation_graph.png')