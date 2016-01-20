# -*- coding: utf-8 -*-

#################################################
# ckip.py
# ckip.py
#
# Copyright (c) 2012-2014, Chi-En Wu
# Distributed under The BSD 3-Clause License
#################################################

from __future__ import unicode_literals

from contextlib import closing
from re import compile
from socket import socket, AF_INET, SOCK_STREAM

from lxml.etree import tostring, fromstring
from lxml.builder import E


# def _construct_parsing_tree(tree_text):
#     parent_node = None
#     current_node = {}
#
#     node_queue = []
#     text = ''
#     is_head = False
#     for char in tree_text:
#         if char == '(':
#             node_queue.append(parent_node)
#
#             current_node['child'] = []
#             current_node['pos'] = text
#             text = ''
#
#             parent_node = current_node
#             current_node = {}
#
#         elif char == ')':
#             if is_head:
#                 parent_node['head'] = current_node
#                 is_head = False
#
#             if len(text) > 0:
#                 current_node['term'] = text
#                 text = ''
#
#             parent_node['child'].append(current_node)
#
#             if is_head:
#                 parent_node['head'] = current_node
#                 is_head = False
#
#             current_node = parent_node
#             parent_node = node_queue.pop()
#
#         elif char == ':':
#             if text == 'Head':
#                 is_head = True
#             else:
#                 current_node['pos'] = text
#
#             text = ''
#
#         elif char == '|':
#             if is_head:
#                 parent_node['head'] = current_node
#                 is_head = False
#
#             if len(text) > 0:
#                 current_node['term'] = text
#                 text = ''
#
#             parent_node['child'].append(current_node)
#             current_node = {}
#
#         else:
#             text += char
#
#     return current_node


class CKIPClient(object):
    _BUFFER_SIZE = 4096
    _ENCODING = 'big5'

    def __init__(self, username, password):
        self.username = username
        self.password = password

    def __build_request_xml(self, text):
        return E.wordsegmentation(
            E.option(showcategory='1'),
            E.authentication(username=self.username, password=self.password),
            E.text(text),
            version='0.1')

    def __send_and_recv(self, msg):
        with closing(socket(AF_INET, SOCK_STREAM)) as s:
            s.connect((self._SERVER_IP, self._SERVER_PORT))
            s.sendall(msg)

            result = ''
            done = False
            while not done:
                chunk = s.recv(self._BUFFER_SIZE)
                result += chunk.decode(self._ENCODING)
                done = result.find('</wordsegmentation>') > -1

        return result

    def _extract_sentence(self, sentence):
        raise NotImplementedError()

    def process(self, text):
        tree = self.__build_request_xml(text)
        msg = tostring(tree, encoding=self._ENCODING, xml_declaration=True)

        result_msg = self.__send_and_recv(msg)
        result_tree = fromstring(result_msg)

        status = result_tree.find('./processstatus')
        sentences = result_tree.iterfind('./result/sentence')
        result = {
            'status': status.text,
            'status_code': status.get('code'),
            'result': [self._extract_sentence(sentence.text)
                       for sentence in sentences]
        }

        return result


class CKIPSegmenter(CKIPClient):
    _SERVER_IP = '140.138.77.225'
    _SERVER_PORT = 1501

    def _extract_sentence(self, sentence):
        pattern = compile('^(.*)\(([^(]+)\)$')
        raw_terms = sentence.split()

        terms = []
        for raw_term in raw_terms:
            match = pattern.match(raw_term)
            term = {
                'term': match.group(1),
                'pos': match.group(2)
            }

            terms.append(term)

        return terms


# class CKIPParser(CKIPClient):
#     _SERVER_IP = '140.109.19.112'
#     _SERVER_PORT = 8000
#
#     def _extract_sentence(self, sentence):
#         pattern = compile('^#\d+:1\.\[0\] (.+)#(.*)$')
#         match = pattern.match(sentence)
#
#         tree_text = match.group(1)
#         tree = _construct_parsing_tree(tree_text)
#
#         raw_punctuation = match.group(2)
#         punctuation = None
#         if len(raw_punctuation) > 0:
#             pattern = compile('^(.*)\(([^(]+)\)$')
#             match = pattern.match(raw_punctuation)
#             punctuation = {
#                 'term': match.group(1),
#                 'pos': match.group(2)
#             }
#
#         result = {
#             'tree': tree,
#             'punctuation': punctuation
#         }
#
#         return result


def segsentence(sentence):
    # from ckip import CKIPParser
    segmenter = CKIPSegmenter('lcyu', 'lcyu')
    # segmenter = CKIPParser('lcyu', 'lcyu')

    if '裏' in sentence:
        sentence = sentence.replace('裏', '裡')
        print('替換了')
        # print(sentence)
    result = segmenter.process(sentence)
    # print(sentence)
    word = ""
    for term_list in result['result']:
        for term in term_list:
            word += " " + term['term']
    return word


if __name__ == '__main__':
    print(segsentence(
        '臺灣電玩競賽隊伍臺北暗殺星TPA)在電玩大賽英雄聯盟世界錦標賽奪冠，電玩產業引起重視，教育部將在3個月之內提出優秀電競選手升學及獎學金相關辦法，激勵今16)日遊戲股全面大漲，擺脫近期低迷的悶氣，由華義3086-TW)領軍攻頂，其中昱泉6169-TW)、網龍3083-TW)隨後亮燈漲停，智冠、宇峻漲幅超過5%，辣椒也都有3%以上漲幅'))

    # Error Texts
    # 臺灣電玩競賽隊伍臺北暗殺星TPA)在電玩大賽英雄聯盟世界錦標賽奪冠，電玩產業引起重視，教育部將在3個月之內提出優秀電競選手升學及獎學金相關辦法，激勵今16)日遊戲股全面大漲，擺脫近期低迷的悶氣，由華義3086-TW)領軍攻頂，其中昱泉6169-TW)、網龍3083-TW)隨後亮燈漲停，智冠5478-TW)、宇峻3546-TW)漲幅超過5%，辣椒4946-TW)、樂升3662-TW)、傳奇4994-TW)、歐買尬3687-TW)、鈊象3293-TW)及橘子6180-TW)也都有3%以上漲幅。
    # 元智大學非常好我想去那裏上學你們怎麼看
