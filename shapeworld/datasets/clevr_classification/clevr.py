from random import randrange
from shapeworld.dataset import Dataset
from shapeworld.datasets import clevr_util


class CLEVRDataset(Dataset):

    dataset_type = 'clevr_classification'
    dataset_name = 'clevr'
    dataset_values = {'world': 'world', 'world_model': 'model', 'question': 'alts(text)', 'question_model': 'alts(model)', 'question_length': 'alts(int)', 'answer': 'alts(int)', 'alternatives': 'int'}

    def __init__(self, directory, parts=dict(train='A', validation='A', test='A')):
        world_size = tuple(next(clevr_util.images_iter(directory=directory, parts=parts, mode='train')).shape[:2])
        self.question_size = 0
        unique_answers = set()
        words = set()
        for _, question, _, answer in clevr_util.questions_iter(directory=directory, parts=parts, mode='train'):
            question = question.split()
            self.question_size = max(self.question_size, len(question))
            words.update(question)
            unique_answers.add(answer)
        words = sorted(words)
        self.answers = sorted(unique_answers)
        self.answers.append('UNKNOWN')
        self.num_answers = len(self.answers)
        super(CLEVRDataset, self).__init__(world_size=world_size, vectors=dict(question=self.question_size), words=words)
        self.clevr = {mode: clevr_util.clevr(directory=directory, parts=parts, mode=mode) for mode in ('train', 'validation', 'test')}

    def specification(self):
        specification = super(CLEVRDataset, self).specification()
        specification['answers'] = self.answers
        return specification

    def generate(self, n, mode=None, noise_range=None, include_model=False, alternatives=False):
        assert noise_range is None
        batch = self.zero_batch(n, include_model=include_model, alternatives=alternatives)
        unknown = self.words['UNKNOWN']
        for i in range(n):
            try:
                world, world_model, questions, question_models, answers = next(self.clevr[mode])
            except StopIteration:
                if i > 0:
                    return {key: value[:i] for key, value in batch.items()}
                else:
                    return None
            batch['world'][i] = world
            if include_model:
                batch['world_model'][i] = world_model
            if alternatives:
                alts = len(questions)
                batch['alternatives'][i] = alts
                batch['question'][i].extend(batch['question'][i][0].copy() for _ in range(alts - 1))
                for a, (question, question_model, answer) in enumerate(zip(questions, question_models, answers)):
                    question = question.split()
                    assert len(question) <= self.question_size
                    for w, word in enumerate(question):
                        batch['question'][i][a][w] = self.words.get(word, unknown)
                    batch['question_length'][i].append(len(question))
                    if include_model:
                        batch['question_model'][i].append(question_model)
                    if answer in self.answers:
                        batch['answer'][i].append(self.answers.index(answer))
                    else:
                        batch['answer'][i].append(self.answers.index('UNKNOWN'))
            else:
                sample = randrange(len(questions))
                question = questions[sample].split()
                assert len(question) <= self.question_size
                for w, word in enumerate(question):
                    batch['question'][i][w] = self.words.get(word, unknown)
                batch['question_length'][i] = len(question)
                if include_model:
                    batch['question_model'][i] = question_models[sample]
                answer = answers[sample]
                if answer in self.answers:
                    batch['answer'][i] = self.answers.index(answer)
                else:
                    batch['answer'][i] = self.answers.index('UNKNOWN')
        return batch


dataset = CLEVRDataset
CLEVRDataset.default_config = dict()
