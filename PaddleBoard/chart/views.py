from django.http import HttpResponse
from django.template import loader
import json


def index(request):
    data = read_chart_data()
    template = loader.get_template('chart/index.html')
    context = {
        'chart_data': json.dumps(data)
    }
    return HttpResponse(template.render(context, request))


def read_chart_data():
    #TODO: READ FROM PROTOBUF
    chart_data = [
        {'x': 1, 'y': 5},
        {'x': 20, 'y': 20},
        {'x': 40, 'y': 10},
        {'x': 60, 'y': 40},
        {'x': 80, 'y': 5},
        {'x': 100, 'y': 60},
        {'x': 120, 'y': 40},
        {'x': 140, 'y': 30}]
    return chart_data
