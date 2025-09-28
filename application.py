from flask_socketio import SocketIO, emit
from flask import Flask, render_template, request
from random import random
from time import sleep
from threading import Thread, Event

from scapy.sendrecv import AsyncSniffer
from flow.Flow import Flow
from flow.PacketInfo import PacketInfo

import numpy as np
import pickle
import csv
import traceback
import json
import pandas as pd
import ipaddress
from urllib.request import urlopen

from tensorflow import keras
from lime import lime_tabular
import dill
import joblib
import plotly
import plotly.graph_objs
import warnings
warnings.filterwarnings("ignore")


def ipInfo(addr=''):
    try:
        url = 'https://ipinfo.io/json' if addr == '' else 'https://ipinfo.io/' + addr + '/json'
        res = urlopen(url)
        data = json.load(res)
        return data['country']
    except Exception:
        return None


__author__ = 'hoang'

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True
socketio = SocketIO(app, async_mode="threading", logger=True, engineio_logger=True)
# Thread and event
thread = Thread()
thread_stop_event = Event()

# CSV logs
f = open("output_logs.csv", 'w', newline='')
w = csv.writer(f)
f2 = open("input_logs.csv", 'w', newline='')
w2 = csv.writer(f2)

# Columns
cols = ['FlowID', 'FlowDuration', 'BwdPacketLenMax', 'BwdPacketLenMin', 'BwdPacketLenMean',
        'BwdPacketLenStd', 'FlowIATMean', 'FlowIATStd', 'FlowIATMax', 'FlowIATMin', 'FwdIATTotal',
        'FwdIATMean', 'FwdIATStd', 'FwdIATMax', 'FwdIATMin', 'BwdIATTotal', 'BwdIATMean', 'BwdIATStd',
        'BwdIATMax', 'BwdIATMin', 'FwdPSHFlags', 'FwdPackets_s', 'MaxPacketLen', 'PacketLenMean',
        'PacketLenStd', 'PacketLenVar', 'FINFlagCount', 'SYNFlagCount', 'PSHFlagCount', 'ACKFlagCount',
        'URGFlagCount', 'AvgPacketSize', 'AvgBwdSegmentSize', 'InitWinBytesFwd', 'InitWinBytesBwd',
        'ActiveMin', 'IdleMean', 'IdleStd', 'IdleMax', 'IdleMin', 'Src', 'SrcPort', 'Dest', 'DestPort',
        'Protocol', 'FlowStartTime', 'FlowLastSeen', 'PName', 'PID', 'Classification', 'Probability',
        'Risk']

ae_features = np.array(['FlowDuration', 'BwdPacketLengthMax', 'BwdPacketLengthMin', 'BwdPacketLengthMean',
                        'BwdPacketLengthStd', 'FlowIATMean', 'FlowIATStd', 'FlowIATMax', 'FlowIATMin',
                        'FwdIATTotal', 'FwdIATMean', 'FwdIATStd', 'FwdIATMax', 'FwdIATMin', 'BwdIATTotal',
                        'BwdIATMean', 'BwdIATStd', 'BwdIATMax', 'BwdIATMin', 'FwdPSHFlags', 'FwdPackets/s',
                        'PacketLengthMax', 'PacketLengthMean', 'PacketLengthStd', 'PacketLengthVariance',
                        'FINFlagCount', 'SYNFlagCount', 'PSHFlagCount', 'ACKFlagCount', 'URGFlagCount',
                        'AveragePacketSize', 'BwdSegmentSizeAvg', 'FWDInitWinBytes', 'BwdInitWinBytes',
                        'ActiveMin', 'IdleMean', 'IdleStd', 'IdleMax', 'IdleMin'])

flow_count = 0
flow_df = pd.DataFrame(columns=cols)
src_ip_dict = {}
current_flows = {}
FlowTimeout = 600

# Load models
ae_scaler = joblib.load("models/preprocess_pipeline_AE_39ft.save")
ae_model = keras.models.load_model('models/autoencoder_39ft.hdf5', compile=False)

with open('models/model.pkl', 'rb') as f_model:
    classifier = pickle.load(f_model)

with open('models/explainer', 'rb') as f_explainer:
    explainer = dill.load(f_explainer)

predict_fn_rf = lambda x: classifier.predict_proba(x).astype(float)


def classify(features):
    global flow_count
    feature_string = [str(i) for i in features[39:]]
    record = features.copy()
    features = [np.nan if x in [np.inf, -np.inf] else float(x) for x in features[:39]]

    if feature_string[0] in src_ip_dict:
        src_ip_dict[feature_string[0]] += 1
    else:
        src_ip_dict[feature_string[0]] = 1

    for i in [0, 2]:
        ip = feature_string[i]
        if not ipaddress.ip_address(ip).is_private:
            country = ipInfo(ip)
            img = f' <img src="static/images/blank.gif" class="flag flag-{country.lower()}" title="{country}">' \
                if country not in [None, 'ano', 'unknown'] else \
                ' <img src="static/images/blank.gif" class="flag flag-unknown" title="UNKNOWN">'
        else:
            img = ' <img src="static/images/lan.gif" height="11px" style="margin-bottom: 0px" title="LAN">'
        feature_string[i] += img

    if np.nan in features:
        return

    result = classifier.predict([features])
    proba = predict_fn_rf([features])
    proba_score = [proba[0].max()]
    proba_risk = sum(list(proba[0, 1:]))
    if proba_risk > 0.8:
        risk = ["<p style=\"color:red;\">Very High</p>"]
    elif proba_risk > 0.6:
        risk = ["<p style=\"color:orangered;\">High</p>"]
    elif proba_risk > 0.4:
        risk = ["<p style=\"color:orange;\">Medium</p>"]
    elif proba_risk > 0.2:
        risk = ["<p style=\"color:green;\">Low</p>"]
    else:
        risk = ["<p style=\"color:limegreen;\">Minimal</p>"]

    classification = [str(result[0])]
    if result != 'Benign':
        print(feature_string + classification + proba_score)

    flow_count += 1
    w.writerow(['Flow #' + str(flow_count)])
    w.writerow(['Flow info:'] + feature_string)
    w.writerow(['Flow features:'] + features)
    w.writerow(['Prediction:'] + classification + proba_score)
    w.writerow(['-' * 100])

    w2.writerow(['Flow #' + str(flow_count)])
    w2.writerow(['Flow info:'] + features)
    w2.writerow(['-' * 100])
    flow_df.loc[len(flow_df)] = [flow_count] + record + classification + proba_score + risk

    ip_data = pd.DataFrame({'SourceIP': list(src_ip_dict.keys()), 'count': list(src_ip_dict.values())}).to_json(orient='records')
    socketio.emit('newresult', {'result': [flow_count] + feature_string + classification + proba_score + risk,
                                "ips": json.loads(ip_data)}, namespace='/test')
    return [flow_count] + record + classification + proba_score + risk


def newPacket(p):
    try:
        packet = PacketInfo()
        packet.setDest(p)
        packet.setSrc(p)
        packet.setSrcPort(p)
        packet.setDestPort(p)
        packet.setProtocol(p)
        packet.setTimestamp(p)
        packet.setPSHFlag(p)
        packet.setFINFlag(p)
        packet.setSYNFlag(p)
        packet.setACKFlag(p)
        packet.setURGFlag(p)
        packet.setRSTFlag(p)
        packet.setPayloadBytes(p)
        packet.setHeaderBytes(p)
        packet.setPacketSize(p)
        packet.setWinBytes(p)
        packet.setFwdID()
        packet.setBwdID()

        if packet.getFwdID() in current_flows:
            flow = current_flows[packet.getFwdID()]
            if (packet.getTimestamp() - flow.getFlowLastSeen()) > FlowTimeout:
                classify(flow.terminated())
                del current_flows[packet.getFwdID()]
                flow = Flow(packet)
                current_flows[packet.getFwdID()] = flow
            elif packet.getFINFlag() or packet.getRSTFlag():
                flow.new(packet, 'fwd')
                classify(flow.terminated())
                del current_flows[packet.getFwdID()]
            else:
                flow.new(packet, 'fwd')
                current_flows[packet.getFwdID()] = flow

        elif packet.getBwdID() in current_flows:
            flow = current_flows[packet.getBwdID()]
            if (packet.getTimestamp() - flow.getFlowLastSeen()) > FlowTimeout:
                classify(flow.terminated())
                del current_flows[packet.getBwdID()]
                flow = Flow(packet)
                current_flows[packet.getFwdID()] = flow
            elif packet.getFINFlag() or packet.getRSTFlag():
                flow.new(packet, 'bwd')
                classify(flow.terminated())
                del current_flows[packet.getBwdID()]
            else:
                flow.new(packet, 'bwd')
                current_flows[packet.getBwdID()] = flow
        else:
            flow = Flow(packet)
            current_flows[packet.getFwdID()] = flow

    except AttributeError:
        return
    except:
        traceback.print_exc()


def snif_and_detect():
    sniffer = AsyncSniffer(prn=newPacket, store=False)
    sniffer.start()
    print("Async sniffer started...")

    try:
        while not thread_stop_event.isSet():
            now = pd.Timestamp.now().timestamp()
            for flow_id, flow in list(current_flows.items()):
                if (now - flow.getFlowLastSeen()) > FlowTimeout:
                    classify(flow.terminated())
                    del current_flows[flow_id]
            sleep(1)
    except Exception as e:
        print("Exception in snif_and_detect:", e)
    finally:
        sniffer.stop()
        print("Sniffer stopped")


@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/flow-detail')
def flow_detail():
    flow_id = request.args.get('flow_id', default=-1, type=int)
    flow = flow_df.loc[flow_df['FlowID'] == flow_id]
    X = [flow.values[0, 1:40]]
    choosen_instance = X
    proba_score = list(predict_fn_rf(choosen_instance))
    risk_proba = sum(proba_score[0][1:])
    if risk_proba > 0.8:
        risk = "Risk: <p style=\"color:red;\">Very High</p>"
    elif risk_proba > 0.6:
        risk = "Risk: <p style=\"color:orangered;\">High</p>"
    elif risk_proba > 0.4:
        risk = "Risk: <p style=\"color:orange;\">Medium</p>"
    elif risk_proba > 0.2:
        risk = "Risk: <p style=\"color:green;\">Low</p>"
    else:
        risk = "Risk: <p style=\"color:limegreen;\">Minimal</p>"

    exp = explainer.explain_instance(choosen_instance[0], predict_fn_rf, num_features=6, top_labels=1)
    X_transformed = ae_scaler.transform(X)
    reconstruct = ae_model.predict(X_transformed)
    err = reconstruct - X_transformed
    abs_err = np.absolute(err)
    ind_n_abs_largest = np.argpartition(abs_err, -5)[-5:]
    col_n_largest = ae_features[ind_n_abs_largest]
    err_n_largest = err[0][ind_n_abs_largest]
    plot_div = plotly.offline.plot({
        "data": [plotly.graph_objs.Bar(x=col_n_largest[0].tolist(), y=err_n_largest[0].tolist())]
    }, include_plotlyjs=False, output_type='div')

    return render_template('detail.html',
                           tables=[flow.reset_index(drop=True).transpose().to_html(classes='data')],
                           exp=exp.as_html(),
                           ae_plot=plot_div,
                           risk=risk)


@socketio.on('connect', namespace='/test')
def test_connect():
    global thread
    print('Client connected')
    if not thread.is_alive():
        print("Starting Sniffer Thread")
        thread = socketio.start_background_task(snif_and_detect)


@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected')


if __name__ == '__main__':
    socketio.run(app)
