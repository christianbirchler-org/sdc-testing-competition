import pymongo
from pymongo.collection import ObjectId
from pymongo.server_api import ServerApi
import psycopg
import mydata
import shapely
import numpy as np
import competition_2026_pb2
import subject

def curvature_profile(test_detail: mydata.TestDetails) -> list[float]:
    """
    Compute the curvature for every meter of the road.

    The following website was used as a reference: https://de.wikipedia.org/wiki/Kr%C3%BCmmung
    """
    #print("compute curvature profile")
    road_shape = shapely.LineString(test_detail.road_points)


    delta_s = 2  # 10 meters

    curvature_profile = np.zeros(int(road_shape.length)) # we want the curvature for every meter
    for s in range(len(curvature_profile)):
        #s = (i+1)*delta_s

        # ignore the edge cases close to the ends of the road
        if (s < delta_s/2) or (s > road_shape.length-delta_s/2):
            continue


        pt_q: shapely.Point = road_shape.interpolate(s-delta_s, normalized=False)
        pt_r: shapely.Point = road_shape.interpolate(s-delta_s/2, normalized=False)

        pt_s: shapely.Point = road_shape.interpolate(s, normalized=False)

        pt_t: shapely.Point = road_shape.interpolate(s+delta_s/2, normalized=False)
        pt_u: shapely.Point = road_shape.interpolate(s+delta_s, normalized=False)

        tangent_r_vec = np.array((pt_s.x-pt_q.x, pt_s.y-pt_q.y))
        tangent_t_vec = np.array((pt_u.x-pt_s.x, pt_u.y-pt_s.y))

        cos_phi = np.dot(tangent_r_vec, tangent_t_vec)/(np.linalg.norm(tangent_r_vec)*np.linalg.norm(tangent_t_vec))
        phi = np.arccos(cos_phi)

        kappa = phi/delta_s
        if np.isnan(kappa):
            continue

        curvature_profile[s] = kappa

    return curvature_profile

def transform_subject_to_testdetails(subj: mydata.Subject) -> mydata.TestDetails:
    pass

def make_test_details_list(raw_test_cases) -> list[mydata.TestDetails]:
    test_details = []
    for raw_test in raw_test_cases:
        test_id = raw_test['_id']['$oid']
        road_points = [(pts['x'], pts['y']) for pts in raw_test['road_points']]
        hasFailed = raw_test['meta_data']['test_info']['test_outcome'] == "FAIL"
        sim_time = raw_test['meta_data']['test_info']['test_duration']
        test_details.append(mydata.TestDetails(test_id=test_id, hasFailed=hasFailed, sim_time=sim_time, road_points=road_points))
    return test_details


mongodb_collections = [
    'campaign_12_freneticV',
    'campaign_2_frenetic',
    'campaign_13_frenetic_v',
    'campaign_14_frenetic',
    'campaign_6_frenetic_v',
    'campaign_9_frenetic',
    'campaign_4_frenetic_v',
    'campaign_6_ambiegen',
    'campaign_11_frenetic',
    'campaign_3_ambiegen',
    'campaign_2_ambiegen',
    'campaign_7_frenetic',
    'campaign_8_ambiegen',
    'campaign_7_frenetic_v',
    'campaign_7_ambiegen',
    'campaign_5_frenetic_v',
    'campaign_9_ambiegen',
    'campaign_11_ambiegen',
    'campaign_4_ambiegen',
    'campaign_4_frenetic',
    'campaign_13_ambiegen',
    'campaign_3_frenetic',
    'campaign_15_ambiegen',
    'campaign_15_freneticV',
    'campaign_11_frenetic_v',
    'campaign_5_frenetic',
    'campaign_14_frenetic_v',
    'campaign_12_frenetic',
    'campaign_15_frenetic',
    'campaign_14_ambiegen',
    'campaign_13_frenetic',
    'campaign_8_frenetic',
    'campaign_5_ambiegen',
    'campaign_6_frenetic',
    'campaign_2_frenetic_v',
    'campaign_10_ambiegen'
]


projection = {
     '_id': 0,
     'sim_time': {'$toDouble': '$OpenDRIVE.header.sdc_test_info.@test_duration'},
     'test_id': {'$toString': '$_id'},
     'hasFailed': {
         '$eq': [
             '$OpenDRIVE.header.sdc_test_info.@test_outcome', 'FAIL'
         ]
     },
     'road_points': {
         '$map': {
             'input': '$OpenDRIVE.road.planView.geometry',
             'as': 'road_points',
             'in': [
                 {
                     '$toDouble': '$$road_points.@x'
                 }, {
                     '$toDouble': '$$road_points.@y'
                 }
             ]
         }
     }
 }


def init_iterator(mongo_client: pymongo.MongoClient,init_data: mydata.Subject):
    """Python generator for the initialization interface for gRPC."""
    sensodat_db = mongo_client.get_database('sensodat')

    for test_case in init_data.pg_test_data:
        mongo_collection = sensodat_db.get_collection(mongodb_collections[test_case.sensodat_collection_id])

        sim_data = mongo_collection.find_one({'_id': ObjectId(test_case.object_id)}, projection)

        road_points = [competition_2026_pb2.RoadPoint(sequenceNumber=i, x=pts[0], y=pts[1]) for i, pts in enumerate(sim_data['road_points'])]
        grpc_test_case = competition_2026_pb2.SDCTestCase(testId=test_case.object_id, roadPoints=road_points)
        oracle: competition_2026_pb2.Oracle = competition_2026_pb2.Oracle(testCase=grpc_test_case, hasFailed=test_case.has_failed)

        yield oracle


def test_suite_iterator(mongo_client: pymongo.MongoClient, subject: mydata.Subject):
    """Python generator for the selection interface for gRPC."""

    sensodat_db = mongo_client.get_database('sensodat')
    for test_case in subject.pg_test_data:
        mongo_collection = sensodat_db.get_collection(mongodb_collections[test_case.sensodat_collection_id])
        sim_data = mongo_collection.find_one({'_id': ObjectId(test_case.object_id)}, projection)
        road_points = [competition_2026_pb2.RoadPoint(sequenceNumber=i, x=pts[0], y=pts[1]) for i, pts in enumerate(sim_data['road_points'])]
        test_case = competition_2026_pb2.SDCTestCase(testId=test_case.object_id, roadPoints=road_points)
        yield test_case
