import os
ROOT_DIR = os.path.join("..")

CLASSIFIED_MODELS = {
    'Industry': [
        {
            'model_file': 'IndustrySGDClassifier_1.pkl',
            'binarizer_file': 'IndustryBinarizer_1.pkl'
        },
    ],
    'Topic': [
        {
            'model_file': 'TopicSGDClassifier_1.pkl',
            'binarizer_file': 'TopicBinarizer_1.pkl'
        }
    ],
    'Reject': [
        {'model_file': 'SI_Non_Business_Reject.pkl'}
    ],
    '135': [
        {
            'model_file': 'custom_tags_model_135_en_1.pkl',
            'binarizer_file': 'custom_tags_binarizer_135_en_1.pkl'
        }
    ],
    '214': [
        {
            'model_file': 'custom_tags_model_214_en_1.pkl',
            'binarizer_file': 'custom_tags_binarizer_214_en_1.pkl'
        }
    ]
}
NER_MODEL = {
    'MITIE': os.path.join(
            ROOT_DIR,
            './nertaggermodel/MITIE-models/english/ner_model.dat'
        )
}
