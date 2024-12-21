import os
import json
import argparse

import numpy as np
import fasttext
import h5py
from tqdm import tqdm
import sklearn
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader

from sum_utils import getlogger, unusedname
from model import SiameseComparator


def load_fasttext(words, fasttext_file):
    ft = fasttext.load_model(fasttext_file)
    demb = ft.get_dimension()
    dctemb = np.zeros((len(words), demb))
    for key, w in enumerate(words):
        dctemb[key] = ft.get_word_vector(w)

    return dctemb


def get_embeddings(vocabfile, embeddingstype, fasttext_file='cc.pt.300.bin'):
    """
    :param vocabfile:
    :param embeddingstype:
    :return: If fasttext, returns an array with the token embeddings, otherwise, the number of tokens.
    """
    with open(vocabfile, "r") as fin:
        vocab = fin.readlines()
        vocab = [w.replace("\n", "") for w in vocab]

    if embeddingstype == 'fasttext':
        logger.info("Loading fastext")
        emb = load_fasttext(vocab, fasttext_file)
    else:
        logger.info("Self trainned embeddings")
        emb = len(vocab)

    return emb


def get_data(filename):
    x1 = []
    x2 = []
    y = []
    with h5py.File(filename, "r") as fin:
        for d1, d2, l in zip(fin["doc1"], fin["doc2"], fin["label"]):
            x1.append(d1)
            x2.append(d2)
            y.append(l)

    y = np.array(y, dtype='f').reshape((-1, 1))
    return np.array(x1), np.array(x2), y


def get_dataloader(x1, x2, y, bsize=32):
    train_data = TensorDataset(torch.tensor(x1), torch.tensor(x2), torch.tensor(y))
    train_sampler = RandomSampler(train_data)
    return DataLoader(train_data, sampler=train_sampler, batch_size=bsize)


def train(model, train_data1, train_data2, train_labels, monitor_data=None, monitor_data2=None, monitor_labels=None,
          epochs=10, patience=3, checkpoint="checkpoint", eval_steps=1000):
    train_dataloader = get_dataloader(train_data1, train_data2, train_labels)
    if monitor_data is not None:
        monit_dataloader = get_dataloader(monitor_data, monitor_data2, monitor_labels)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = torch.nn.BCELoss()

    # Tracking best validation accuracy
    best_loss = np.inf
    patience_count = 0
    eval_count = 0
    eval_steps = min(eval_steps, len(train_dataloader))

    model.train()
    for epoch_i in range(epochs):
        epoch_loss = 0

        dloader = tqdm(train_dataloader, desc="Epoch {}".format(epoch_i), dynamic_ncols=True, leave=False)
        for step, batch in enumerate(dloader):
            eval_count += 1
            x1_batch, x2_batch, y_batch = tuple(t for t in batch)

            # Clean gradientes
            optimizer.zero_grad()

            # Feed the model
            y_pred = model(x1_batch, x2_batch)

            # Loss calculation
            loss = criterion(y_pred, y_batch)

            epoch_loss += loss.item()

            # Gradients calculation
            loss.backward()
            del loss
            del y_pred

            # Gradients update
            optimizer.step()

            if eval_count % eval_steps == eval_steps - 1:
                currloss = epoch_loss / (step + 1)
                patience_count += 1

                # Evaluation phase
                if monitor_data is not None:
                    metrics = evaluate(model, monit_dataloader)
                    logger.info(str(currloss) + " " + json.dumps(metrics))
                    currloss = metrics['loss']

                if best_loss > currloss:
                    logger.info("Saving best model {}, on epoch {}, step {}".format(currloss, epoch_i, step + 1))
                    best_loss = currloss
                    torch.save({
                        'vocab_len': model.emb.num_embeddings,
                        'model_state_dict': model.state_dict(),
                        'emb_dim': model.emb.embedding_dim,
                        'docemb_dim': model.fc_in.out_features,
                        'nconv': model.conv2.out_channels,
                    }, checkpoint)

                    patience_count = 0

                if patience_count == patience:
                    logger.info("Early stop on epoch {}, step {}".format(epoch_i, step + 1))
                    break
        else:
            continue
        break


def evaluate(model, dataloader):
    trainning = model.training
    model.eval()
    y_expected = None
    y_predicted = None
    total_loss = 0

    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc="Evaluating", position=0, dynamic_ncols=True, leave=False)):
            x1_batch, x2_batch, y_batch = tuple(t for t in batch)

            y_pred = model(x1_batch, x2_batch)
            total_loss += torch.nn.functional.binary_cross_entropy(y_pred, y_batch).numpy().item()

            y_batch = y_batch.numpy()
            y_expected = y_batch if y_expected is None else np.append(y_expected, y_batch, axis=0)

            y_pred = torch.round(y_pred).detach().numpy()
            y_predicted = y_pred if y_predicted is None else np.append(y_predicted, y_pred, axis=0)

    f1 = f1_score(y_expected, y_predicted)
    acc = accuracy_score(y_expected, y_predicted)
    prec = precision_score(y_expected, y_predicted)
    recl = recall_score(y_expected, y_predicted)

    if trainning:
        model.train()

    return {
        'f1': f1,
        'acc': acc,
        'precision': prec,
        'recall': recl,
        'loss': total_loss / (step+1)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data', type=str, help='Trainning input data file')
    parser.add_argument('vocab', type=str, help='List of words. Line number is the index.')
    parser.add_argument('--test_data', type=str, help='Model final evaluation')
    parser.add_argument('--validation', type=str, default="0.1",
                        help='Validation file or fraction (in decimal) of the training set for validation.')
    parser.add_argument('--embeddings', choices=['self', 'fasttext'], default='fasttext', help='Word embeddings')
    parser.add_argument('--fasttext_file', type=str, default='cc.en.300.bin', help='FastText model file')
    parser.add_argument('--freeze_embedding', action='store_true', help='Add to freeze input embeddings')
    parser.add_argument('--embedding_size', type=int, default=300, help='Embeddings layer size, when embeddings=self')
    parser.add_argument('--nconv', type=int, default=128, help='Number of convolutional units per windows size')
    parser.add_argument('--docemb', type=int, default=32, help='Document representation size')
    parser.add_argument('--checkpoint', type=str, default="checkpoint.pt", help='checkpoint output file')
    parser.add_argument('--overwritecheckpoint', action='store_true', help='overwrite previous checkpoint files')
    parser.add_argument('--eval_steps', type=int, default=300, help='Training steps between evaluations')
    parser.add_argument('--max_epochs', type=int, default=10, help='Maximum number of trainning epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size on each trainning epoch')
    parser.add_argument('--logfile', type=str, default="model_trainning.log", help='Output log file')
    parser.add_argument('--seed', type=int, default=0)


    args = parser.parse_args()

    logger = getlogger(args.logfile)
    logger.info("Filename: " + os.path.basename(__file__))
    logger.info(args)

    checkpointname = args.checkpoint

    if args.train_data:
        emb = get_embeddings(args.vocab, args.embeddings, args.fasttext_file)

        try:
            valpart = float(args.validation)
            data_x1, data_x2, data_y = get_data(args.train_data)
            x1, x1_val, x2, x2_val, y, y_val = sklearn.model_selection.train_test_split(
                data_x1, data_x2, data_y, test_size=valpart, stratify=data_y, random_state=args.seed)
        except ValueError:
            x1, x2, y = get_data(args.train_data)
            x1_val, x2_val, y_val = get_data(args.validation)

        model = SiameseComparator(embeddings=emb, emb_dim=args.embedding_size, freeze_emb=args.freeze_embedding,
                                  nconv=args.nconv, docemb_dim=args.docemb)
        logger.info(model)

        if not args.overwritecheckpoint:
            checkpointname = unusedname(args.checkpoint)
            if checkpointname != args.checkpoint:
                logger.info("Checkpoint changed to: " + checkpointname)

        train(model, x1, x2, y, x1_val, x2_val, y_val, checkpoint=checkpointname, eval_steps=args.eval_steps,
              epochs=args.max_epochs)

    if args.test_data:
        test_x1, test_x2, test_y = get_data(args.test_data)

        checkpoint = torch.load(checkpointname)
        model = SiameseComparator(
            embeddings=checkpoint['vocab_len'],
            emb_dim=checkpoint['emb_dim'],
            nconv=checkpoint['nconv'],
            docemb_dim=checkpoint['docemb_dim']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        test_dataloader = get_dataloader(test_x1, test_x2, test_y, args.batch_size)
        metrics = evaluate(model, test_dataloader)
        logger.info(json.dumps(metrics, indent=4))
