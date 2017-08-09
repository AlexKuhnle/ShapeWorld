from random import randrange
from shapeworld.dataset import Dataset
from shapeworld.datasets import clevr_util


class CLEVRDataset(Dataset):

    dataset_type = 'clevr_answering'
    dataset_name = 'clevr'
    dataset_values = {'world': 'world', 'world_model': 'model', 'question': 'alts(text)', 'question_model': 'alts(model)', 'question_length': 'alts(int)', 'answer': 'alts(text)', 'answer_length': 'alts(int)', 'alternatives': 'int'}

    def __init__(self, directory, parts=None):
        world_size = tuple(next(clevr_util.images_iter(directory=directory, parts=parts, mode='train')).shape[:2])
        self.question_size = 0
        self.answer_size = 0
        words = set()
        for _, question, _, answer in clevr_util.questions_iter(directory=directory, parts=parts, mode='train'):
            question = question.split()
            answer = answer.split()
            self.question_size = max(self.question_size, len(question))
            self.answer_size = max(self.answer_size, len(answer))
            words.update(question)
            words.update(answer)
        words = sorted(words)
        super(CLEVRDataset, self).__init__(world_size=world_size, vectors=dict(question=self.question_size, answer=self.answer_size), words=words)
        self.clevr = {mode: clevr_util.clevr(directory=directory, parts=parts, mode=mode) for mode in ('train', 'validation', 'test')}

    def generate(self, n, mode=None, noise_range=None, include_model=False, alternatives=False):
        assert noise_range is None or noise_range == 0.0
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
                batch['answer'][i].extend(batch['answer'][i][0].copy() for _ in range(alts - 1))
                for a, (question, question_model, answer) in enumerate(zip(questions, question_models, answers)):
                    question = question.split()
                    assert len(question) <= self.question_size
                    for w, word in enumerate(question):
                        batch['question'][i][a][w] = self.words.get(word, unknown)
                    batch['question_length'][i].append(len(question))
                    if include_model:
                        batch['question_model'][i].append(question_model)
                    answer = answer.split()
                    assert len(answer) <= self.answer_size
                    for w, word in enumerate(answer):
                        batch['answer'][i][a][w] = self.words.get(word, unknown)
                    batch['answer_length'][i].append(len(answer))
            else:
                sample = randrange(len(questions))
                question = questions[sample].split()
                assert len(question) <= self.question_size
                for j, word in enumerate(question):
                    batch['question'][i][j] = self.words.get(word, unknown)
                batch['question_length'][i] = len(question)
                if include_model:
                    batch['question_model'][i] = question_models[sample]
                answer = answers[sample].split()
                assert len(answer) <= self.answer_size
                for j, word in enumerate(answer):
                    batch['answer'][i][j] = self.words.get(word, unknown)
                batch['answer_length'][i] = len(answer)
        return batch


dataset = CLEVRDataset
CLEVRDataset.default_config = dict()
