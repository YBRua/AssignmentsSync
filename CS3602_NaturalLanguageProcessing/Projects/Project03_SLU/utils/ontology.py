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
        # self.pinyinchar2sems: Dict[str, Dict[str, set]] = {}
        if self.pinyin:
            self.pinyin2sems: Dict[str, Dict[str, str]] = {}
            # self.sems2pinyin: Dict[str, Dict[str, str]] = {}
        else:
            self.pinyin2sems = None
            # self.sems2pinyin = None

        for slot, values in ontology_json['slots'].items():
            char2sem = defaultdict(set)
            # pinyinchar2sem = defaultdict(set)
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
                # self.sems2pinyin[slot] = sem2pinyin

            # construct character-to-sentence map to speed up searching
            for semantic in self.ontology[slot]:
                for char in semantic:
                    if self.pinyin:
                        pinyin = pypinyin.lazy_pinyin(char)
                        assert len(pinyin) == 1, f"Huh? {len(pinyin)}"
                        pinyin = pinyin[0]
                        # pinyinchar2sem[pinyin].add(semantic)
                    char2sem[char].add(semantic)
            self.char2sems[slot] = char2sem
            # self.pinyinchar2sems[slot] = pinyinchar2sem

            self.edcache = {}

    def _pinyinization(self, s: str):
        return ''.join(pypinyin.lazy_pinyin(s))

    def _load_data_from_path(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = f.read(-1).splitlines()

        return data

    # def _get_semantic_by_pinyin(self, slot, value):
    #     closest_dist = 1919810
    #     candidate_sem = UNK
    #     tok2sem = self.pinyinchar2sems[slot]

    #     candidates = set.union(*[tok2sem[self._pinyinization(c)] for c in value])
    #     candidates = set([self.sems2pinyin[slot][cand] for cand in candidates])

    #     # shortcut
    #     if value in candidates:
    #         return self.pinyin2sems[slot][value], 0

    #     # search for closest value in ontology
    #     for candidate in candidates:
    #         dist = self._normalized_distance(value, candidate)
    #         if closest_dist > dist:
    #             closest_dist = dist
    #             candidate_sem = candidate

    #     # convert pinyin back to characters
    #     if self.pinyin:
    #         if candidate_sem == UNK:
    #             return candidate_sem, closest_dist
    #         candidate_sem = self.pinyin2sems[slot][candidate_sem]

    #     return candidate_sem, closest_dist

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
        return sem, dist
        # if (sem == UNK or dist > thres) and self.pinyin:
        #     return self._get_semantic_by_pinyin(slot, self._pinyinization(value))
        # else:
        #     return sem, dist

    def _normalized_distance(self, s1: str, s2: str) -> float:
        denominator = max(len(s1), len(s2))
        if self.pinyin:
            edit_dist = self._pinyin_weighted_levenshtein_distance(s1, s2)
        else:
            edit_dist = self._ordinary_levenshtein_distance(s1, s2)

        return edit_dist / denominator

    def _character_pinyin_distance(self, char1, char2):
        if char1 + char2 in self.edcache:
            return self.edcache[char1 + char2]
        elif char2 + char1 in self.edcache:
            return self.edcache[char2 + char1]
        pinyin1 = self._pinyinization(char1)
        pinyin2 = self._pinyinization(char2)
        dist = self._ordinary_levenshtein_distance(pinyin1, pinyin2)
        dist = dist / max(len(pinyin1), len(pinyin2)) * 2.5

        self.edcache[char1 + char2] = dist
        # print(char1, char2, dist)
        return dist

    def _ordinary_levenshtein_distance(self, s1, s2):
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
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]

    def _pinyin_weighted_levenshtein_distance(self, s1, s2) -> int:
        # from Wikipedia article; Iterative with two matrix rows.
        if s1 == s2:
            return 0
        elif len(s1) == 0:
            return len(s2)
        elif len(s2) == 0:
            return len(s1)

        v0 = [None] * (len(s2) + 1)
        v1 = [None] * (len(s2) + 1)

        for i in range(len(v0)):
            v0[i] = i

        for i in range(len(s1)):
            v1[0] = i + 1
            for j in range(len(s2)):
                cost = 0 if s1[i] == s2[j] else self._character_pinyin_distance(s1[i], s2[j])
                v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
            for j in range(len(v0)):
                v0[j] = v1[j]

        return v1[len(s2)]


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
