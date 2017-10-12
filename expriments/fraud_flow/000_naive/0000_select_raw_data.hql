# /home/work/lib/hadoop-hive/hive/bin/hive -e ""

SELECT baiduid, accesstimestamp, method, urlpre, clientip, statuscode, referer, useragent
FROM wap_hao123_access
WHERE dt = '20171010' AND hr >= 08 and hr <= 12;