# Kvasir-SEG

The experiment runner obtains the dataset from [Simula](https://datasets.simula.no/kvasir-seg/). Download the [official archive](https://datasets.simula.no/downloads/kvasir-seg.zip) to `data/kvasir-seg.zip` to run without a dataset download.

The runner verifies SHA-256 `03b30e21d584e04facf49397a2576738fd626815771afbbf788f74a7153478f7` and extracts only its image and mask directories. Filenames are paired by stem and dataset copies and split lists are ignored by Git. The aggregate split fingerprint and counts are included in the result evidence. The synthetic smoke command generates its own data locally.
