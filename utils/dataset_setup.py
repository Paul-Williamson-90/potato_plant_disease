import random
import os
import pandas as pd

data_dir = os.path.abspath("potato-plant-diseases-data/PlantVillage/PlantVillage")

if __name__=="__main__":
    
    classes = os.listdir(data_dir)

    dataset = pd.DataFrame(columns=["image", "label"])
    for i, c in enumerate(classes):
        images = os.listdir(os.path.join(data_dir, c))
        dataset = pd.concat(
            [
                dataset, 
                pd.DataFrame({"image": [f"{c}/{x}" for x in images], "label": [i] * len(images)})
            ], 
            axis=0
        ).reset_index(drop=True)

    dataset["split"] = None
    idxs = dataset.index.tolist()
    random.shuffle(idxs)
    n = len(idxs)
    train_size = int(0.8 * n)
    test_size = n - train_size
    dataset.loc[idxs[:train_size], "split"] = "train"
    dataset.loc[idxs[train_size:], "split"] = "test"

    csv_path = os.path.abspath("potato-plant-diseases-data/PlantVillage/dataset_metadata.csv")
    dataset.to_csv(csv_path, index=False)

    yaml = f"""image_meta_path: {csv_path}
image_data_path: {data_dir}"""
    
    with open("config.yaml", "w") as f:
        f.write(yaml)

