from random import randrange
from shapeworld import util
from shapeworld.dataset import Dataset
from shapeworld.datasets import clevr_util


class ClevrDataset(Dataset):

    def __init__(self, directory, parts=None, pixel_noise_stddev=None):
        values = dict(world='world', world_model='model', question='alternatives(language)', question_length='alternatives(int)', question_model='alternatives(model)', answer='alternatives(int)', alternatives='int')
        if parts is not None:
            parts = parts.split(',')
        world_size = tuple(next(clevr_util.images_iter(directory=directory, parts=parts, mode='train')).shape[:2])
        self.question_size = 0
        unique_answers = set()
        vocabulary = set()
        for _, question, _, answer in clevr_util.questions_iter(directory=directory, parts=parts, mode='train'):
            self.question_size = max(self.question_size, len(question))
            vocabulary.update(question)
            assert len(answer) == 1
            unique_answers.add(answer[0])
        vocabularies = dict(language=sorted(vocabulary))
        self.answers = sorted(unique_answers)
        self.answers.append('[UNKNOWN]')
        self.num_answers = len(self.answers)
        super(ClevrDataset, self).__init__(values=values, world_size=world_size, pixel_noise_stddev=pixel_noise_stddev, vectors=dict(question=self.question_size), vocabularies=vocabularies)
        self.clevr = {mode: clevr_util.clevr(directory=directory, parts=parts, mode=mode) for mode in ('train', 'validation', 'test')}

    @property
    def name(self):
        return 'clevr'

    @property
    def type(self):
        return 'clevr_classification'

    def specification(self):
        specification = super(ClevrDataset, self).specification()
        specification['answers'] = self.answers
        return specification

    def generate(self, n, mode=None, noise_range=None, include_model=False, alternatives=False):
        batch = self.zero_batch(n, include_model=include_model, alternatives=alternatives)
        vocabulary = self.vocabularies['language']
        unknown = vocabulary['[UNKNOWN]']
        for i in range(n):
            try:
                world, world_model, questions, question_models, answers = next(self.clevr[mode])
            except StopIteration:
                if i > 0:
                    return {key: value[:i] for key, value in batch.items()}
                else:
                    return None
            batch['world'][i] = self.apply_pixel_noise(world=world)
            if include_model:
                batch['world_model'][i] = world_model
            if alternatives:
                alts = len(questions)
                batch['alternatives'][i] = alts
                batch['question'][i].extend(batch['question'][i][0].copy() for _ in range(alts - 1))
                for a, (question, question_model, answer) in enumerate(zip(questions, question_models, answers)):
                    assert len(question) <= self.question_size
                    for w, word in enumerate(question):
                        batch['question'][i][a][w] = vocabulary.get(word, unknown)
                    batch['question_length'][i].append(len(question))
                    if include_model:
                        batch['question_model'][i].append(question_model)
                    assert len(answer) == 1
                    answer = answer[0]
                    if answer in self.answers:
                        batch['answer'][i].append(self.answers.index(answer))
                    else:
                        batch['answer'][i].append(self.answers.index('[UNKNOWN]'))
            else:
                sample = randrange(len(questions))
                assert len(questions[sample]) <= self.question_size
                for w, word in enumerate(questions[sample]):
                    batch['question'][i][w] = vocabulary.get(word, unknown)
                batch['question_length'][i] = len(questions[sample])
                if include_model:
                    batch['question_model'][i] = question_models[sample]
                answer = answers[sample]
                if answer in self.answers:
                    batch['answer'][i] = self.answers.index(answer)
                else:
                    batch['answer'][i] = self.answers.index('[UNKNOWN]')
        return batch

    def get_html(self, generated, image_format='bmp', image_dir=''):
        id2word = self.vocabulary(value_type='language')
        questions = generated['question']
        question_lengths = generated['question_length']
        answers = generated['answer']
        data_html = list()
        for n, (question, question_length, answer) in enumerate(zip(questions, question_lengths, answers)):
            data_html.append('<div class="instance"><div class="world"><img src="{image_dir}world-{world}.{format}" alt="world-{world}.{format}"></div><div class="num"><p><b>({num})</b></p></div><div class="questions">'.format(image_dir=image_dir, world=n, format=image_format, num=(n + 1)))
            for question, question_length, answer in zip(question, question_length, answer):
                data_html.append('<p>{question}&ensp;&ndash;&ensp;{answer}</p>'.format(
                    question=util.tokens2string(id2word[word] for word in question[:question_length]),
                    answer=self.answers[answer]
                ))
            data_html.append('</div></div>')
        html = '<!DOCTYPE html><html><head><title>{dtype} {name}</title><style>.data{{width: 100%; height: 100%;}} .instance{{width: 100%; margin-top: 1px; margin-bottom: 1px; background-color: #DDEEFF;}} .world{{height: {world_height}px; display: inline-block; vertical-align: middle;}} .num{{display: inline-block; vertical-align: middle; margin-left: 10px;}} .questions{{display: inline-block; vertical-align: middle; margin-left: 10px;}}</style></head><body><div class="data">{data}</div></body></html>'.format(
            dtype=self.type,
            name=self.name,
            world_height=self.world_shape[0],
            data=''.join(data_html)
        )
        return html
