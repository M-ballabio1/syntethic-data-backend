from sdv.datasets.demo import download_demo
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata

metadata = SingleTableMetadata()
print(metadata)

data, metadata = download_demo(
    modality='single_table',
    dataset_name='adult'
)

#train

synthesizer = TVAESynthesizer(
    metadata, # required
    enforce_min_max_values=True,
    enforce_rounding=False,
    epochs=3
)
synthesizer.fit(data)
synthesizer.get_loss_values()

synthesizer.save(
    filepath='my_synthesizer.pkl')


# evaluation
synthesizer = TVAESynthesizer.load(
    filepath='my_synthesizer.pkl'
)

synthetic_data = synthesizer.sample(num_rows=10)
print(synthetic_data)

# save the data as a CSV
synthetic_data.to_csv('synthetic_data.csv', index=False)