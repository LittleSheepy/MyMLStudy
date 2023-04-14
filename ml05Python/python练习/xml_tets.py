import dicttoxml
from xml.dom.minidom import parseString
import os

d = {
    "filename":"82236457.jpg",
    "object":{"name":"20220410",
               "bndbox":{
                    "xmin":1961,
                    "ymin":1513,
                    "xmax":2612,
                    "ymax":1651
               }
          },
    "object":{"name":"20220410",
               "bndbox":{
                    "xmin":2612,
                    "ymin":1513,
                    "xmax":2712,
                    "ymax":1651
               }
          }
}
bxml = dicttoxml.dicttoxml(d,custom_root='annotation')
xml = bxml.decode('utf-8')
print(xml)
print('----------------------')
dom = parseString(xml)
prettyxml = dom.toprettyxml(indent='    ')
print(prettyxml)

#将XML字符串保存到文件中。
os.makedirs('files',exist_ok=True)
f=open('82236457.xml','w',encoding='utf-8')
f.write(prettyxml)
f.close()