#!/usr/bin/env python
# coding: utf-8

# In[39]:


import xml.sax


# In[43]:


def str2position(str):
    list = str.split(',');
    pos = [round(float(list[0])), round(float(list[1]))]
    return pos
class MovieHandler( xml.sax.ContentHandler ):
    def __init__(self):
        self.images = []
        self.current_image = ''
        self.labels = {
            '147560': 'Boeing',
            '147655': 'B-1',
            '148026': 'B-2',
            '153105': 'A-10',
            '153163': 'A-26',
            '153203': 'B-29',
            '154799': 'B-52',
            '156147': 'C-5',
            '156184': 'C-17',
            '156667': 'C-21'
        }
        self.seqs = {
            '147560': '0',
            '147655': '1',
            '148026': '2',
            '153105': '3',
            '153163': '4',
            '153203': '5',
            '154799': '6',
            '156147': '7',
            '156184': '8',
            '156667': '9'
        }

    # 元素开始事件处理
    def startElement(self, tag, attributes):
        if(tag == 'image'):
            self.current_image = {
                'name': attributes['name'],
                'seq': self.seqs[attributes['task_id']], 
                'label': self.labels[attributes['task_id']], 
                'task_id': attributes['task_id'],
                'points': None
            }
        elif(tag == 'points'):
            self.current_image['points'] = list(map(str2position, attributes['points'].split(';')))
            self.images.append(self.current_image)
            
    # 元素结束事件处理
    def endElement(self, tag):
        if(tag == 'image'):
            current_image = None
    
    # 内容事件处理
    def characters(self, content):
        pass
    
# In[44]:


def getLabels(url="/Users/jeffxie/Desktop/Aircraft Recognition/data-set/annotations_10.xml"):
    # 创建一个 XMLReader
    parser = xml.sax.make_parser()
    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
 
    # 重写 ContextHandler
    Handler = MovieHandler()
    parser.setContentHandler(Handler)
   
    parser.parse(url)
    return Handler.images

