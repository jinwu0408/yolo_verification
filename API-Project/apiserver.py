#!/usr/bin/python3
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from json import dumps

db_connect = create_engine('sqlite:///soterdb.db')
app = Flask(__name__)
api = Api(app)




class get_Drone_Task(Resource):
    def get(self,drone_id):
        conn = db_connect.connect()
        query = conn.execute("select * from tasks where DroneId ="+ drone_id +" order by taskid")
        result = {'data': [dict(zip(tuple (query.keys()) ,i)) for i in query.cursor]}
        return jsonify(result)


class get_Drone_Photo(Resource):
    def get(self,drone_id):
        conn = db_connect.connect()
        query = conn.execute("select * from photos where droneid ="+ drone_id +"")
        result = {'data': [dict(zip(tuple (query.keys()) ,i)) for i in query.cursor]}
        return jsonify(result)


class set_Drone_Photo(Resource):
    def get(self,Photo_path):
        conn = db_connect.connect()
        x = Photo_path.split("$")
        strPhoto_path=x[0]
        strdroneId=x[1]
        realx1=x[2]
        realy1=x[3]
        realx2=x[4]
        realy2=x[5]
        strlabel=x[6]
        realscore=x[7]
        print(realscore)
        print(strPhoto_path)
        strPhoto_path=strPhoto_path.replace("|","/")
        print(strPhoto_path)
        print(strdroneId)


        query = conn.execute("insert into photos(photo_path,droneid,x1,y1,x2,y2,label, confidence_score) values('" + str(strPhoto_path) + "',"+ strdroneId +","+ realx1 +","+ realy1 +","+ realx2 +","+ realy2 +","+ strlabel +","+ realscore +")")

       # result = {'data': [dict(zip(tuple (query.keys()) ,i)) for i in query.cursor]}
        return "done"


class delete_data_with_id(Resource):
   def get(self,drone_id):
       conn = db_connect.connect()
       query = conn.execute("delete from photos where droneid = {}".format(drone_id))
       return 'deleted'


class update_confidence_score(Resource):
   def get(self,confidence_score):
        conn = db_connect.connect()
        x = confidence_score.split("$")
        strPhoto_path=x[0]
        realscore=x[1]
        print (realscore)
        print (strPhoto_path)
        strPhoto_path=strPhoto_path.replace("|","/")
        query = conn.execute("update photos set confidence_score="+ realscore +"  where photo_path='" + str(strPhoto_path) + "'" )
        return "updated"


api.add_resource(get_Drone_Task, '/task/<drone_id>')
api.add_resource(set_Drone_Photo, '/set_Drone_Photo/<Photo_path>')
api.add_resource(get_Drone_Photo, '/get_Drone_Photo/<drone_id>')
api.add_resource(update_confidence_score, '/update_confidence_score/<confidence_score>')
api.add_resource(delete_data_with_id, '/delete/<drone_id>')


if __name__ == '__main__':
     app.run()
