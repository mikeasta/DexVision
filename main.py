from dexnet.train import train
from dexnet.evaluate import evaluate_on_images

TRAIN = True


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    if TRAIN:
        train("data/pokemons/", epochs=50)
    else:
        evaluate_on_images(input_dir="data/input",
                           output_dir="data/output")
