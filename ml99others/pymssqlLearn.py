# -*- coding:utf-8 -*-
"""
@author:Administrator
"""
import pymssql


def main():
    # 创建SqlServer连接
    conn = pymssql.connect(host='BJ-DZ0102871\MSSQLSERVER01', user='sa', password='wms', database='test')
    if conn:
        print("连接成功!")
    # 如果和本机数据库交互，只需修改链接字符串
    #conn=pymssql.connect(host='.',database='test')
    # 创建游标
    cur = conn.cursor()
    cur.execute('select top 5 * FROM [dbo].[Resident]')
    # 如果update/delete/insert记得要conn.commit()
    # 否则数据库事务无法提交
    print(cur.fetchall())
    # 关闭游标
    cur.close()
    # 关闭连接
    conn.close()


if __name__ == '__main__':
    main()
