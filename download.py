from allennlp_models import pretrained

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    pretrained.load_predictor("coref-spanbert")

if __name__ == "__main__":
    download_model()
