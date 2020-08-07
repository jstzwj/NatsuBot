from peewee import *
import datetime


db = SqliteDatabase('chat.db')
# db = MySQLDatabase('qqbot', user='root', password='', host='localhost', port=3306)

class BaseModel(Model):
    class Meta:
        database = db

class Chat(BaseModel):
    user = TextField(null=True)
    group = TextField(null=True)
    kind = TextField() # 'g': group, 't': temp, 'f': friend
    message = TextField()
    date = DateTimeField(default=datetime.datetime.now)


if __name__ == "__main__":
    db.connect()
    db.create_tables([Chat])
