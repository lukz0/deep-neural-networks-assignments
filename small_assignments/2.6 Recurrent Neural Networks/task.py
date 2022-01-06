import pickle
import random
import torch.nn as nn
import torch
import zipfile
import csv
import io
import itertools
import re
from torch.utils.tensorboard import SummaryWriter
from bs4 import BeautifulSoup

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#tags: dict[int, list[str]] = {}
## Id: [OwnerUserID, CreationDate, Score, Title, Body]
#questions = {}
## ID: [OwnerUserId, CreationDate, ParentId, Score, Body]
#answers = {}
#
# with zipfile.ZipFile('archive.zip', 'r') as zf:
#    with zf.open('Tags.csv', 'r') as f:
#        reader = csv.reader(io.TextIOWrapper(
#            f, encoding='utf-8', errors='ignore'))
#
#        # Skip header
#        reader = iter(reader)
#        next(reader)
#        for elem in reader:
#            key = int(elem[0])
#            taglist: list[str] = tags.get(key, [])
#            taglist.append(elem[1])
#            tags[key] = taglist
#        tags = {item[0]: item[1].sort() or ' '.join(item[1]) for item in tags.items()}
#
#    with zf.open('Questions.csv', 'r') as f:
#        reader = csv.reader(io.TextIOWrapper(
#            f, encoding='utf-8', errors='ignore'))
#        #print(*itertools.islice(reader, 100), sep='\n\n')
#
#        # Skip header
#        reader = iter(reader)
#        next(reader)
#
#        questions = {int(elem[0]): elem[1:] for elem in reader}
#
#    with zf.open('Answers.csv', 'r') as f:
#        reader = csv.reader(io.TextIOWrapper(
#            f, encoding='utf-8', errors='ignore'))
#        #print(*itertools.islice(reader, 100), sep='\n\n')
#
#        # Skip header
#        reader = iter(reader)
#        next(reader)
#
#        answers = {int(elem[0]): elem[1:] for elem in reader}
#
#import pickle
#
# with open('cache.pickle', 'wb') as f:
#    pickle.dump({'tags': tags, 'questions': questions, 'answers': answers}, f)
with open('cache.pickle', 'rb') as f:
    global tags
    global questions
    global answers
    cache = pickle.load(f)
    tags = cache['tags']
    questions = cache['questions']
    answers = cache['answers']
    print('Cache loaded')
# list of answer keys for random sampling
## ID: [OwnerUserId, CreationDate, ParentId, Score, Body]
#answers = {}
answers: dict[int, list[str]] = answers
# Remove all answers with negative score
answers: dict[int, list[str]] = {key: value for key, value in answers.items() if int(value[3]) >= 0}
answers_keys = list(answers.keys())


INVALIDCHARACTER = re.compile(
    r'[^a-zA-Z0-9\?\!\.\,\:\; \t\n\>\<\=\}\{\[\]\(\)\%\&\\/\"\'\+\-\_\*\~\^\|\§ \t\n`\#\¤\&\@\^\$]')
#
#
# def alltagsgen():
#    for tags_val in tags.values():
#        for tag in tags_val.split(' '):
#            yield tag
#
#
#alltags: list[str] = set(alltagsgen())
#
#
# def alltagcharactersgen():
#    for tags_val in tags.values():
#        for tag_char in tags_val:
#            yield tag_char
#
#
#alltagcharacters: list[str] = list(set(alltagcharactersgen()))
#
#
# def allquestioncharactersgen():
#    for question in questions.values():
#        for character in re.sub(INVALIDCHARACTER, u'\uFFFD', question[4]):
#            yield character
#
#
#allquestioncharacters = list(set(allquestioncharactersgen()))
#
#
# def allanswercharactersgen():
#    for answer in answers.values():
#        for character in re.sub(INVALIDCHARACTER, u'\uFFFD', answer[4]):
#            yield character
#
#
#allanswercharacters = list(set(allanswercharactersgen()))
#allanswercharacters_indexes = {char: i for i, char in enumerate(allanswercharacters)}
#
# all_characters = list(set(itertools.chain(
#    alltagcharacters, allanswercharacters, allquestioncharacters)))
all_characters = ['\t', '\n', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                  'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', '¤', '§', u'\uFFFD']
all_characters_indexes = {char: i for i, char in enumerate(all_characters)}
n_characters = len(all_characters)
#print('All tags:')
#print(*alltags, sep='\n')
#print('\nLen all tags:')
# print(len(alltags))

#print('\nLen all tag characters:')
# print(len(alltagcharacters))

# print(allanswercharacters)

#print('\nLen all answer characters:')
# print(len(allanswercharacters))


#file = ""
# with open('movie_lines.txt') as f:
#    file = unidecode.unidecode(f.read())


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size: int, num_layers: int, output_size: int):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        out = self.embed(x)
        out, (hidden, cell) = self.lstm(out.unsqueeze(1), (hidden, cell))
        out = self.fc(out.reshape(out.shape[0], -1))
        return out, (hidden, cell)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size,
                             self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers, batch_size,
                           self.hidden_size).to(device)
        return hidden, cell


class Generator():
    def __init__(self):
        # self.chunk_len = 2500  # 250
        self.num_epochs = 50000#5000
        self.batch_size = 1
        self.print_every = 50
        self.hidden_size = 256 * 3
        self.num_layers = 2
        self.lr = 0.003
        self.rnn = RNN(n_characters, self.hidden_size,
                       self.num_layers, n_characters).to(device)
        self.rnn.load_state_dict(torch.load('model20500.bin'))

    def char_tensor(self, string):
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            tensor[c] = all_characters_indexes.get(
                string[c], all_characters_indexes[u'\uFFFD'])
            #tensor[c] = all_characters.index(string[c])
        return tensor

    # def get_random_batch_old(self):
    #    start_idx = random.randint(0, len(file) - self.chunk_len)
    #    end_idx = start_idx + self.chunk_len + 1
    #    text_str = file[start_idx:end_idx]
    #    text_input = torch.zeros(self.batch_size, self.chunk_len)
    #    text_target = torch.zeros(self.batch_size, self.chunk_len)
#
    #    for i in range(self.batch_size):
    #        text_input[i, :] = self.char_tensor(text_str[:-1])
    #        text_target[i, :] = self.char_tensor(text_str[1:])
#
    #    return text_input.long(), text_target.long()

    def get_random_batch(self):
        text_chunk = get_random_QA()
        chunk_len = len(text_chunk)-1
        text_input = torch.zeros(self.batch_size, chunk_len)
        text_target = torch.zeros(self.batch_size, chunk_len)

        for i in range(self.batch_size):
            text_input[i, :] = self.char_tensor(text_chunk[:-1])
            text_target[i, :] = self.char_tensor(text_chunk[1:])

        return chunk_len, text_input.long(), text_target.long()

    def generate(self, initial_str='TITLE:', prediction_len=10000, temperature=0.5):
        hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)
        initial_input = self.char_tensor(initial_str)
        predicted = initial_str

        for p in range(len(initial_str) - 1):
            _, (hidden, cell) = self.rnn(
                initial_input[p].view(1).to(device), hidden, cell)

        last_char = initial_input[-1]

        for p in range(prediction_len):
            output, (hidden, cell) = self.rnn(
                last_char.view(1).to(device), hidden, cell)
            output_dist = output.data.view(-1).div(temperature).exp()
            top_char = torch.multinomial(output_dist, 1)[0]
            predicted_char = all_characters[top_char]
            predicted += predicted_char
            last_char = self.char_tensor(predicted_char)
            # If END_OF_ANSWER is generated, exit sooner
            if predicted.endswith('END_OF_ANSWER'):
                break

        return predicted

    def train(self):
        # input_size, hidden_size, num_layers, output_size
        self.rnn = RNN(n_characters, self.hidden_size,
                       self.num_layers, n_characters).to(device)
        self.rnn.load_state_dict(torch.load('model16000.bin'))
        optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.lr, eps=1e-3, amsgrad=True)

        optimizer.load_state_dict(torch.load('optimizer16000.bin'))
        #optimizer = torch.optim.AdamW(
        #    self.rnn.parameters(), lr=self.lr, eps=1e-3, amsgrad=True, weight_decay=1e-6)
        criterion = nn.CrossEntropyLoss().to(device)

        # writer = SummaryWriter(f'runs/names0')  # for tensorboard

        print("=> Starting training")

        with open('1.log', 'at') as log:
            for epoch in range(16001, self.num_epochs + 1):
                chunk_size, input, target = self.get_random_batch()
                hidden, cell = self.rnn.init_hidden(self.batch_size)

                self.rnn.zero_grad()
                loss = 0
                input = input.to(device)
                target = target.to(device)

                for c in range(chunk_size):
                    output, (hidden, cell) = self.rnn(input[:, c], hidden, cell)
                    loss += criterion(output, target[:, c])

                loss.backward()
                optimizer.step()
                loss = loss.item() / chunk_size

                if epoch % self.print_every == 0:
                    print(f'Epoch: {epoch}\tLoss: {loss}')
                    log.write(f'Epoch: {epoch}\tLoss: {loss}\n')
                    log.flush()
                    print(self.generate())
                    if epoch % 500 == 0:
                        torch.save(self.rnn.state_dict(), f'model{epoch}.bin')
                        torch.save(optimizer.state_dict(), f'optimizer{epoch}.bin')

            #writer.add_scalar('Training loss', loss, global_step=epoch)


NON_BREAKABLE_SPACE_RE = re.compile(r'\xa0')


def get_random_QA():
    #tags: dict[int, list[str]] = {}
    ## Id: [OwnerUserID, CreationDate, Score, Title, Body]
    #questions = {}
    ## ID: [OwnerUserId, CreationDate, ParentId, Score, Body]
    #answers = {}
    global questions
    global answers
    random_answer = answers[random.sample(answers_keys, 1)[0]]
    question = questions[int(random_answer[2])]
    question_body = re.sub(
        NON_BREAKABLE_SPACE_RE,
        ' ',
        BeautifulSoup(question[4], 'html.parser').get_text()
    ).strip(' ')
    question_title = re.sub(
        NON_BREAKABLE_SPACE_RE,
        ' ',
        BeautifulSoup(question[3], 'html.parser').get_text()
    ).strip(' ')
    answer_body = re.sub(
        NON_BREAKABLE_SPACE_RE,
        ' ',
        BeautifulSoup(random_answer[4], 'html.parser').get_text()
    ).strip(' ')

    return f'TITLE:\n{question_title}\nQUESTION_BODY:\n{question_body}\nANSWER_BODY:\n{answer_body}\nEND_OF_ANSWER'


generator = Generator()
#generator.train()

#print('Savind the rnn')
#torch.save(generator.rnn.state_dict(), 'model.bin')

while(True):
    title = input('Question Title: ')
    body = input('Question Body: ')

    print(generator.generate(initial_str=f'TITLE:\n{title}\nQUESTION_BODY:\n{body}\nANSWER_BODY:\n'))
