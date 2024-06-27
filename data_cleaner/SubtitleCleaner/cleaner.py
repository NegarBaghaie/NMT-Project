# ADD LICENSE
import re
from srttime import SrtTime
import csv
import sys


def main():
    if len(sys.argv) != 3:
        en_file = input('English srt filename >> ')
        fa_file = input('Persian srt filename >> ')
    else:
        _, en_file, fa_file = sys.argv

    en_subtitles = get_subtitles(en_file)
    fa_subtitles = get_subtitles(fa_file)

    max_len = len(en_subtitles)
    en_start = 0
    fa_start = 0

    en_end = en_start + 1
    fa_end = fa_start + 1

    lst = []
    prev_op = 'n'
    options = True
    while True:
        en_texts = en_subtitles[en_start:en_end]
        fa_texts = fa_subtitles[fa_start:fa_end]
        en_texts = [en['text'] for en in en_texts]
        fa_texts = [fa['text'] for fa in fa_texts]
        en_text = ' '.join(en_texts)
        fa_text = ' '.join(fa_texts)
        print(f'en:{en_end} fa:{fa_end} out of {max_len}\nENGLISH: {en_text}\nFARSI: {fa_text}')

        if options:
            str_inp = "'es' skip\n'n' next both\n's' to save\n'm' merge with previous"
            if fa_end <= len(fa_subtitles):
                str_inp += "\n'f' next farsi"
            if en_end <= len(en_subtitles):
                str_inp += "\n'e' next english"
            if len(en_texts) > 0:
                str_inp += "\n're' pop english"
            if len(fa_texts) > 0:
                str_inp += "\n'rf' pop farsi"
            if options:
                str_inp += "\n'hide' to hide the options menu"
            else:
                str_inp += "\n'show' to show the options menu"

            if len(lst) > 0:
                str_inp += "\n'R' remove previously added item"
            str_inp += "\n'END' Save all and write to file"
            str_inp += '\n>> '
            inp = input(str_inp)
        else: inp = input('\n>>')

        if inp == 'es':
            en_start = en_end
            fa_start = fa_end
            en_end = en_start + 1
            fa_end = fa_start + 1
            prev_op = 'es'
        elif inp == 'm' and len(lst) > 0:
            lst[-1]['en'] += f' {en_text}'
            lst[-1]['fa'] += f' {fa_text}'
            res_en = lst[-1]['en']
            res_fa = lst[-1]['fa']
            print('MERGE result:')
            print(f'ENGLISH: {res_en}\nFARSI: {res_fa}\n\n')
            en_start = en_end
            fa_start = fa_end
            en_end = en_start + 1
            fa_end = fa_start + 1
            prev_op = 'm'
        elif inp == 'n':
            en_end += 1
            fa_end += 1
            prev_op = 'n'
        elif inp == 's':
            lst.append({'en': en_text, 'fa': fa_text})
            en_start = en_end
            fa_start = fa_end
            en_end = en_start + 1
            fa_end = fa_start + 1
            prev_op = 's'
        elif inp == 'f' and fa_end <= len(fa_subtitles):
            fa_end += 1
            prev_op = 'n'
        elif inp == 'e' and en_end <= len(en_subtitles):
            en_end += 1
            prev_op = 'n'
        elif inp == 're' and len(en_texts) > 0:
            en_popped = en_texts.pop()
            print(f'Item popped: {en_popped}\n\n')
            en_end -= 1
            prev_op = 're'
        elif inp == 'rf' and len(fa_texts) > 0:
            fa_popped = fa_texts.pop()
            print(f'Item popped: {fa_popped}\n\n')
            fa_end -= 1
            prev_op = 'rf'
        elif inp == 'R' and len(lst) > 0:
            item = lst.pop()
            en_popped = item['en']
            fa_popped = item['fa']
            print(f'Item Removed: {en_popped} : {fa_popped}\n\n')
            prev_op = 'R'
        elif inp == 'hide':
            options = False
        elif inp == 'show':
            options = True
        elif inp == 'END':
            break

        if en_end > len(en_subtitles) and fa_end > len(fa_subtitles):
            if prev_op == 'n':
                lst.append({'en': en_text, 'fa': fa_text})
            break

    with open('output.csv', 'w') as file:
        writer = csv.writer(file)
        for sample in lst:
            writer.writerow(list(sample.values()))

    print('File saved. Checkout \'output.csv\'')


def get_subtitles(filename: str) -> list:
    with open(filename, 'r', encoding='utf8') as file:
        file = file.read()

    pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})((\n.+)*)'
    matches = re.findall(pattern, file)
    subtitles = []
    for match in matches:
        index = int(match[0])
        start = SrtTime.getStrTime(match[1])
        end = SrtTime.getStrTime(match[2])
        lines = match[3].split('\n')[1:]
        text = ' '.join(lines).strip()
        subtitles.append({'index': index, 'start': start, 'end': end, 'text': text})

    return subtitles


def get_subtitles_up_to(subtitles: list,start:SrtTime, upto:SrtTime) -> list:
    lst = []
    for sub in subtitles:
        if start <= sub['start'] < upto:
            lst.append(sub)
    return lst


if __name__ == "__main__":
    main()
