import os
import time
import abc
import sys
import json
import argparse

import subject
import sampling

import metrics
import experimental_setup

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import psycopg

import competition_2026_pb2
import competition_2026_pb2_grpc

import grpc
import shapely
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import pairwise_distances
from pymongo import MongoClient
from pymongo.collection import Collection, ObjectId


if __name__ == "__main__":
    # load environment variables fro .env file
    load_dotenv()
    sensodat_uri = os.getenv('SENSODAT_URI')
    postgres_uri = str(os.getenv('POSTGRES_URI'))

    # handle CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--url")
    args = parser.parse_args()

    if args.url:
        GRPC_URL = args.url
    else:
        print('provide url to the tool with -u/--url')
        sys.exit(1)


    me = metrics.MetricEvaluator()
    mongo_client = MongoClient(sensodat_uri)
    with psycopg.connect(postgres_uri) as pg_conn:
        tl = sampling.CompetitionEvaluationSampler(mongo_client, subject.OverallRandomSubjectCreationStrategy())
        es = experimental_setup.ExperimentalSetup(pg_conn, mongo_client, me, tl)

        # set up gRPC conection stub
        channel = grpc.insecure_channel(GRPC_URL)
        stub = competition_2026_pb2_grpc.CompetitionToolStub(channel)  # stub represents the tool

        experimental_results = es.start_experiment(stub=stub, sample_size=50, subject_size=300)
        print(experimental_results)
