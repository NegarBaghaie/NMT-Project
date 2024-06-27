#!/usr/bin/env python3
import os
import time

from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import threading

BATCH_SIZE = 20
MAX_THREADS = 5


def main():
    try:
        translator = GoogleTranslator(source='en', target='fa')
    except:
        translator = None

    filename = input('filename >> ')
    sep = input('SEP >> ')

    if translator is None:
        print('Could not load the translator !!!')
        return

    new_filename = input('Filename to save to >> ')
    rest_filename = input('Save the undone (blank to use the original file) >> ')

    if rest_filename.strip() == '':
        rest_filename = filename

    with open(filename, 'r') as file:
        texts = file.read().split('\n')

    total_batches = []
    i = 0
    for batch in range(len(texts) // BATCH_SIZE):
        batch_list = []
        while len(batch_list) != BATCH_SIZE:
            batch_list.append(texts[i].split(sep=sep)[0])
            i += 1
        total_batches.append(batch_list)

    batch_count = len(total_batches)

    taken = []
    batches_done = []
    checkpoint = 0
    cnt = 0
    for en_batch, fa_batch, batch_id in batch_executor(total_batches, translator):
        print(f'\rBatch {cnt + 1:>6} out of {batch_count:>6}', end='')
        for i in range(len(en_batch)):
            en, fa = en_batch[i], fa_batch[i]
            if len(en.strip()) < 3: continue
            taken.append(sep.join([en, fa]))
            batches_done.append(batch_id)

        if cnt >= checkpoint:
            save(new_filename, rest_filename, taken, total_batches, batches_done)
            checkpoint = cnt + (0.01 * len(total_batches) // BATCH_SIZE)
            print('\nCheckpoint')

        cnt += 1

    save(new_filename, rest_filename, taken, total_batches, batches_done)
    return


def translate_batch(batch, translator, batch_id=0):
    try:
        google = translator.translate_batch(batch, timeout=BATCH_SIZE * 5)
        return batch, google, batch_id
    except:
        print('\nEstablishing connection . . .')
        return batch, None, batch_id


def batch_executor(batches, translator):
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [executor.submit(translate_batch,
                                   batch=batches[i],
                                   translator=translator,
                                   batch_id=i)
                   for i in range(len(batches))]

        for future in as_completed(futures):
            result = future.result()
            if result[1] is None:
                futures.append(executor.submit(translate_batch,
                                               batch=result[0],
                                               translator=translator,
                                               batch_id=result[2]))
                print(f'Batch id {result[2]} has been postponed')
            else:
                yield result


def save(new_filename, rest_filename, taken, total_batches, batches_done):
    if new_filename.strip() == '':
        new_filename = 'output.txt'
    with open(new_filename, 'w') as file:
        file.write('\n'.join(taken))

    if len(batches_done) < len(total_batches):
        if rest_filename.strip() == '':
            return
        with open(rest_filename, 'w') as file:
            for batch_id in range(len(total_batches)):
                if batch_id not in batches_done: continue
                for data in total_batches[batch_id]:
                    file.write(data + '\n')


def thread_sleep(seconds):
    event = threading.Event()
    event.wait(timeout=seconds)


if __name__ == '__main__':
    main()