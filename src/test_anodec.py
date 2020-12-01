import anodec as ano
if __name__=='__main__':
    dirFeatex = "../pretrained_model/featex_spliced_250.h5"
    dirAno = "../pretrained_model/anodec_spliced_250.h5"
    model = ano.load_anodec(dirFeatex, dirAno)
    model.summary()