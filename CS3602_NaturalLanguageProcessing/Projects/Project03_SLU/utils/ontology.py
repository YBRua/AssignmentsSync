import os
import json
import pypinyin
from typing import List, Dict, Tuple
from collections import defaultdict

from utils.vocab import UNK

EPS = 1e-7


class Ontology():
    def __init__(self, dataroot: str, pinyin: bool = False) -> None:
        ontology_path = os.path.join(dataroot, 'ontology.json')
        ontology_json = json.load(open(ontology_path, 'r', encoding='utf-8'))

        self.pinyin = pinyin
        self.ontology: Dict[str, List[str]] = {}
        self.char2sems: Dict[str, Dict[str, set]] = {}
        self.pinyinchar2sems: Dict[str, Dict[str, set]] = {}
        if self.pinyin:
            self.pinyin2sems: Dict[str, Dict[str, str]] = {}
            self.sems2pinyin: Dict[str, Dict[str, str]] = {}
        else:
            self.pinyin2sems = None
            self.sems2pinyin = None

        for slot, values in ontology_json['slots'].items():
            char2sem = defaultdict(set)
            pinyinchar2sem = defaultdict(set)
            pinyin2sem = {}
            sem2pinyin = {}

            # load text file if value is a relative path
            if isinstance(values, str):
                fpath = os.path.join(dataroot, values)
                values = self._load_data_from_path(fpath)  # List[str]

            # convert list to sets
            self.ontology[slot] = set(values)

            # pinyinization
            if self.pinyin:
                for semantic in self.ontology[slot]:
                    pinyin = self._pinyinization(semantic)
                    pinyin2sem[pinyin] = semantic
                    sem2pinyin[semantic] = pinyin
                self.pinyin2sems[slot] = pinyin2sem
                self.sems2pinyin[slot] = sem2pinyin

            # construct character-to-sentence map to speed up searching
            for semantic in self.ontology[slot]:
                for char in semantic:
                    if self.pinyin:
                        pinyin = pypinyin.lazy_pinyin(char)
                        assert len(pinyin) == 1, f"Huh? {len(pinyin)}"
                        pinyin = pinyin[0]
                        pinyinchar2sem[pinyin].add(semantic)
                    char2sem[char].add(semantic)
            self.char2sems[slot] = char2sem
            self.pinyinchar2sems[slot] = pinyinchar2sem

    def _pinyinization(self, s: str):
        return '-'.join(pypinyin.lazy_pinyin(s))

    def _load_data_from_path(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = f.read(-1).splitlines()

        return data

    def _get_semantic_by_pinyin(self, slot, value):
        closest_dist = 1919810
        candidate_sem = UNK
        tok2sem = self.pinyinchar2sems[slot]

        candidates = set.union(*[tok2sem[self._pinyinization(c)] for c in value])
        candidates = set([self.sems2pinyin[slot][cand] for cand in candidates])

        # shortcut
        if value in candidates:
            return self.pinyin2sems[slot][value], 0

        # search for closest value in ontology
        for candidate in candidates:
            dist = self._normalized_distance(value, candidate)
            if closest_dist > dist:
                closest_dist = dist
                candidate_sem = candidate

        # convert pinyin back to characters
        if self.pinyin:
            if candidate_sem == UNK:
                return candidate_sem, closest_dist
            candidate_sem = self.pinyin2sems[slot][candidate_sem]

        return candidate_sem, closest_dist

    def _get_semantic_by_character(self, slot, value):
        closest_dist = 1919810
        candidate_sem = UNK
        char2sem = self.char2sems[slot]

        candidates = set.union(*[char2sem[c] for c in value])

        if value in candidates:
            # print('Char hit')
            return value, 0

        # search for closest value in ontology
        for candidate in candidates:
            dist = self._normalized_distance(value, candidate)
            if closest_dist > dist:
                closest_dist = dist
                candidate_sem = candidate

        return candidate_sem, closest_dist

    def get_closest_semantic(self, slot, value, thres) -> Tuple[str, int]:
        slot = slot.split('-')[-1]
        sem, dist = self._get_semantic_by_character(slot, value)
        if (sem == UNK or dist > thres) and self.pinyin:
            return self._get_semantic_by_pinyin(slot, self._pinyinization(value))
        else:
            return sem, dist

    def _normalized_distance(self, s1: str, s2: str) -> float:
        denominator = max(len(s1), len(s2))
        edit_dist = self._pairwise_levenshtein_distance(s1, s2)

        return edit_dist / denominator

    def _pairwise_levenshtein_distance(self, s1, s2) -> int:
        # from <https://stackoverflow.com/questions/2460177/edit-distance-in-python>
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2+1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(
                        1 + min((
                            distances[i1],
                            distances[i1 + 1],
                            distances_[-1]
                        ))
                    )
            distances = distances_
        return distances[-1]


if __name__ == '__main__':
    import sys

    install_path = os.path.abspath(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    sys.path.append(install_path)

    from utils.vocab import UNK

    dataroot = './data'
    ontology = Ontology(dataroot, True)

    test1 = ontology.get_closest_semantic('终点目标', '川东骨科医院')
    print(test1)
    test2 = ontology.get_closest_semantic('请求类型', '你个笨蛋')
    print(test2)
