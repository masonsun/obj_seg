from models.crfrnn_model import get_crfrnn_model_def
import utils

_INPUT_FILENAME = "image.jpg"
_OUTPUT_FILENAME = "labels.png"

# downloaded from https://goo.gl/ciEYZi
_MODEL_WEIGHTS = "crfrnn_keras_model.h5"

def main():
    model = get_crfrnn_model_def()
    model.load_weights(_MODEL_WEIGHTS)

    img_data, img_h, img_w = utils.get_preprocessed_image(_INPUT_FILENAME)
    probs = model.predict(img_data, verbose=False)[0, :, :, :]
    segmentation = utils.get_label_image(probs, img_h, img_w)
    segmentation.save(_OUTPUT_FILENAME)

if __name__ == "__main__":
    main()
