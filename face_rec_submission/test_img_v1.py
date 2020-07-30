# load an image and predict the class
def run_example():
        # load the inv mapping
        im_file = open(sys.argv[2],'r')
        inv_mapping = json.load(im_file)
        print(inv_mapping)
        # load the image
        img = load_image(sys.argv[3])
        # load model
        model = load_model(sys.argv[1], custom_objects={"fbeta":fbeta})
        # predict the class
        result = model.predict(img)
        print(result[0])
        # map prediction to tags
        tags = prediction_to_tags(inv_mapping, result[0])
        print(tags)

run_example()
