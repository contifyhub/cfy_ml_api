import itertools
from collections import defaultdict
from functools import lru_cache

import joblib
import uvicorn
from mitie import named_entity_extractor
from pathlib import Path

from constants import CLASSIFIED_MODELS, NER_MODEL
from pydantic import BaseModel
from fastapi import Depends, FastAPI, HTTPException, status
import os
import secrets
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import config

@lru_cache()
def get_settings():
    return config.Settings()


app = FastAPI()
security = HTTPBasic()
ROOT_DIR = os.path.join("..")
BASE_PATH = "{}/classified_data/".format(ROOT_DIR)
classified_model_map = defaultdict(lambda: defaultdict(list))
ner_model = dict()


def get_auth_status(
        credentials: HTTPBasicCredentials = Depends(security),
        settings: config.Settings = Depends(get_settings)):

    correct_username = secrets.compare_digest(
        credentials.username, settings.api_username
    )
    correct_password = secrets.compare_digest(
        credentials.password, settings.api_password
    )
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return 200


for model_type in CLASSIFIED_MODELS:
    for model_dict in CLASSIFIED_MODELS[model_type]:
        model_file = model_dict.get('model_file')
        if not model_file:
            continue
        classified_model_map[model_type]['models'].append(joblib.load(
            os.path.join(BASE_PATH, os.path.join(model_type, model_file))
        ))

        binarizer_file = model_dict.get('binarizer_file')
        if binarizer_file:
            classified_model_map[model_type]['binarizer'].append(joblib.load(
                os.path.join(BASE_PATH,
                             os.path.join(model_type, binarizer_file))
            ))

for model_type in NER_MODEL:
    if model_type == 'MITIE':
        ner_model[model_type] = named_entity_extractor(
            NER_MODEL[model_type])


def predict_classes(model_map, text_list, multilabel=False):
    """This function is used to predict classes.

       params: model_map
               text_list
               multilabel:Default(False)
       Return: predictions
       """
    prediction = list()
    classification_models = model_map['models']
    binarizer_model = model_map.get('binarizer', [None])
    for classification_model, binarizer_model in zip(
            *[classification_models, binarizer_model]):
        pred = classification_model.predict(text_list)
        if multilabel:
            pred = binarizer_model.inverse_transform(pred)
        else:
            pred = pred.tolist()
        prediction.append(pred)
    if multilabel:
        final_predictions = [list(map(int, list(itertools.chain(*p))))
                             for p in zip(*prediction)]
    else:
        final_predictions = [list(pred_tup) for pred_tup in zip(*prediction)]
    return final_predictions


class NerText(BaseModel):
    text: list


class StoryText(BaseModel):
    text: list

@app.post('/predict/mitie/')
async def predict_mitie(story: NerText,
                        auth_status: int = Depends(get_auth_status)):

    """This api is used to extract entities from Text using mitie.
    params: story: NerText
    Return: predictions
    """
    data = story.dict()
    text_token = data['text']
    mitie = ner_model['MITIE']
    entities = mitie.extract_entities(text_token)
    _entities = [(list(ent_tup[0]), ent_tup[1], ent_tup[2])
                 for ent_tup in entities]

    return {
        'result': _entities
    }


@app.post('/predict/topic/')
async def predict_topics(story: StoryText,
                         auth_status: int = Depends(get_auth_status)):
    """This api is used to predict Topic from Text.

    params: story: StoryText
    Return: predictions
    """
    data = story.dict()
    text_list = data['text']
    topic_list = classified_model_map['Topic']
    predictions = predict_classes(topic_list, text_list, multilabel=True)
    return {
        'result': predictions
    }


@app.post('/predict/industry/')
async def predict_industry(story: StoryText,
                           auth_status: int = Depends(get_auth_status)):
    """This api is used to predict Industry entities from Text.
    params: story: StoryText
    Return: predictions
    """

    data = story.dict()
    text_list = data['text']
    industry_model_list = classified_model_map['Industry']
    predictions = predict_classes(industry_model_list, text_list,
                                  multilabel=True)
    return {
        'result': predictions
    }

@app.post('/predict/customtags/{client_id}/')
async def predict_ct(client_id: str, story: StoryText,
                     auth_status: int = Depends(get_auth_status)):
    """This api is used to attach Customtags to StoryText.

    params: story: StoryText
            client_id: id of the client
    Return: predictions
    """
    data = story.dict()
    text_list = data['text']
    ct_models_list = classified_model_map[client_id]
    predictions = predict_classes(ct_models_list, text_list, multilabel=True)
    return {
        'result': predictions
    }


@app.post('/predict/reject/')
async def predict_ct(story: StoryText,
                     auth_status: int = Depends(get_auth_status)):
    """This api is used to  reject StoryText.

    params: story: StoryText
    Return: predictions
    """

    data = story.dict()
    text_list = data['text']
    ct_models_list = classified_model_map['Reject']
    predictions = predict_classes(ct_models_list, text_list)
    return {
        'result': predictions
    }


if __name__ == '__main__':
    uvicorn.run(f"{Path(__file__).stem}:app", port=8080, host='localhost',
                reload=True)
