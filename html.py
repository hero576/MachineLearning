import re
import urllib.request


with open('d:/user/gm/desktop/12343.txt','r',encoding='utf8') as f:
    content=f.read()

    ret = re.findall(r'<img alt="" src="(.*?)"',content)

    for i,url in enumerate(ret):
        with open('d:/user/gm/desktop/1/%d.png'%i,'wb') as f2:
            f2.write( urllib.request.urlopen(url).read())

