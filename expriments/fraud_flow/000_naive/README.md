人工智障一号
============

## Features

|特征|计算方法|
|---|-----|
|1，PV | 总请求量/1000，大于1000的是1 |
|2，click num / PV | urlpre为j.php 的请求量/总请求量 |
|3，4xx percent | statuscode=4xx的请求量/总请求量 |
|4，Referer | referer不为空的请求量/总请求量 |
|5，query_interval | 两个请求之间间隔时间的平均值/一小时 |
|6，Head/(Get+Head) | HEAD的请求量/总请求量 |
|7，PV(index+channel)/PV | pv=(urlpre 不是hao123_api 或者hao123_app的请求量),max(pv(i))/pv |
|8，PV(hao123_api)/PV | urlpre为hao123_api的请求量/总请求量 |
