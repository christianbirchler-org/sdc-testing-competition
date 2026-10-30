import dataclasses

import click
import grpc
import numpy as np
import pymongo
import shapely

import interface_2027_pb2 as interface
import interface_2027_pb2_grpc as server_client


@click.group()
def cli():
    click.echo("")

@cli.command()
def name():
    click.echo("start tool evaluator")

    channel = grpc.insecure_channel("localhost:50051")
    tool = server_client.CompetitionToolStub(channel=channel)

    tool_name_reply: interface.NameReply = tool.Name(interface.Empty())

    click.echo("name of the tool: {}".format(tool_name_reply.name))


@cli.command()
@click.argument("test-executions", required=False, help="list of already executed test cases")
@click.option("--sensodat-uri", help="sensodat uri")
def initialize(test_executions, sensodat_uri):
    click.echo("initializing")

    channel = grpc.insecure_channel('localhost:50051')
    stub = server_client.CompetitionToolStub(channel=channel)

    mongo_client = pymongo.MongoClient(host=sensodat_uri)
    sensodat = mongo_client.get_database('sensodat')

    collection = sensodat.get_collection('campaign_11_frenetic')
    test_executions_iterator = _get_test_executions(collection)

    initialization_reply: interface.InitializationReply = stub.Initialize(test_executions_iterator)
    if initialization_reply.ok:
        click.echo('success')
    else:
        click.echo('fail')


@cli.command(help="returns the index of the closest roadpoint to the OOB. Returns -1 on error.")
@click.option("--sensodat-uri", default="mongodb://localhost:27017", help="sensodat uri")
@click.option("--sensodat-collection", default='campaign_11_frenetic', help="sensodat collection name")
@click.option("--evaluation", is_flag=True, default=False, required=False, help="evaluation report")
def minimize(sensodat_uri, sensodat_collection, evaluation):
    click.echo("minimizing")

    channel = grpc.insecure_channel('localhost:50051')
    stub = server_client.CompetitionToolStub(channel=channel)

    mongo_client = pymongo.MongoClient(host=sensodat_uri)
    sensodat = mongo_client.get_database('sensodat')

    collection = sensodat.get_collection(sensodat_collection)

    test_executions_iterator = _get_test_executions(collection)

    for test_execution in test_executions_iterator:
        minimization_reply: interface.MinimizationReply = stub.Minimize(test_execution.testCase)
        std_out_str = "something went wrong"
        if evaluation:
            evaluation_report = Evaluator.evaluate(minimization_reply, test_execution, oob=0.5)
            click.echo(evaluation_report)
        else:
            std_out_str = "test_id: {}\npoint_id: {}".format(minimization_reply.testId, minimization_reply.roadPoint)
            click.echo(std_out_str)


@dataclasses.dataclass
class EvaluationReport:
    test_id: str
    observed_oob_id: int
    estimated_oob_id: int


class Evaluator:
    @staticmethod
    def evaluate(answer: interface.MinimizationReply, test_execution: interface.SDCTestExecution, oob: float) -> EvaluationReport:
        return EvaluationReport(
                test_id=answer.testId,
                observed_oob_id=_get_oob_index(test_execution, oob),
                estimated_oob_id=answer.roadPoint
                )


def _get_oob_index(test_execution: interface.SDCTestExecution, oob: float) -> int:
    """
    Compute the index at what road point the OOB happened.
    :return: Index of OOB, or -1 if there is not OOB (i.e., the car stayed in-lane)
    """
    road_points = [(roadPoint.x, roadPoint.y) for roadPoint in test_execution.testCase.roadPoints]
    center_line: shapely.LineString = shapely.LineString(coordinates=road_points)
    right_lane = center_line.buffer(distance=-5, single_sided=True)
    ideal_trajectory: shapely.LineString | shapely.MultiLineString = center_line.parallel_offset(distance=2.5)
    
    road_model = {
        'road_points': road_points,
        'center_line': center_line,
        'right_lane': right_lane,
        'ideal_trajectory': ideal_trajectory
    }

    trajectory_points = [(position.x, position.y) for position in test_execution.trajectory.trajectoryPoints]
    for position in trajectory_points:
        if out_of_bounds(pos=position, rm=road_model, oob=oob):
            oob_index, _ = closest_point(pos=position, road_pts=road_points)
            return oob_index

    return -1


def out_of_bounds(pos: tuple, rm: dict, oob: float) -> bool:
    """
    Compute of the car at the current position is out of bounds (OOB) according to the OOB parameter.
    :param pos: The current position of the car
    :param rm: Dictionary containing road information
    :param oob: Out of bounds parameter
    :return: True if OOB, False otherwise
    """
    car_model = shapely.box(pos[0]-1, pos[1]-1, pos[0]+1, pos[1]+1)
    diff = car_model.difference(rm['right_lane'])
    actual_oob = diff.area / car_model.area
    return actual_oob > oob

def closest_point(pos: tuple, road_pts: list[tuple[float, float]]) -> tuple[int, float]:
    """
    This function is purely functional.
    :param pos: 
    :param road_pts: 
    :return: Return the index of the closest road point to the given position coordinates.
    """
    pos_np = np.array(pos)
    curr_min_dist = -1
    curr_min_dist_idx = -1
    for i, road_pt in enumerate(road_pts):
        road_pt_np = np.array(road_pt)
        norm = np.linalg.norm(road_pt_np - pos_np)
        if norm < curr_min_dist or curr_min_dist == -1:
            curr_min_dist = norm
            curr_min_dist_idx = i

    return curr_min_dist_idx, curr_min_dist


def _get_test_cases(collection):
    for doc in collection.aggregate(pipeline=SENSODAT_QUERY_PIPELINE):
        sdc_test_case = _get_SDCTestCase(doc)
        yield sdc_test_case

def _get_test_id(doc: dict) -> str:
    return doc['meta_data']['test_info']['test_id']

def _get_SDCTestCase(doc: dict) -> interface.SDCTestCase:
    road_points= [interface.RoadPoint(
        sequenceNumber=ind,
        x=point['x'],
        y=point['y'],
    ) for ind, point in enumerate(doc['road_points'])]
    return interface.SDCTestCase(testId=_get_test_id(doc), roadPoints=road_points)


def _get_test_executions(collection):
    for doc in collection.aggregate(pipeline=SENSODAT_QUERY_PIPELINE):
        sdc_test_execution = _get_SDCTestExecution(doc)
        yield sdc_test_execution


def _get_SDCTestExecution(doc: dict) -> interface.SDCTestExecution:
    road_points= [interface.RoadPoint(
        sequenceNumber=ind,
        x=point['x'],
        y=point['y'],
    ) for ind, point in enumerate(doc['road_points'])]
    
    test_case = interface.SDCTestCase(testId=_get_test_id(doc), roadPoints=road_points)

    trajectory_points = [interface.TrajectoryPoint(
        sequenceNumber=ind,
        x=entry['x'],
        y=entry['y'],
    ) for ind, entry in enumerate(doc['trajectory'])]

    return interface.SDCTestExecution(
                hasFailed=doc['meta_data']['test_info']['test_outcome'] != "PASS",
                testCase=test_case,
                trajectory=interface.SDCTrajectory(testId=doc['meta_data']['test_info']['test_id'], trajectoryPoints=trajectory_points)
            )


SENSODAT_QUERY_PIPELINE = [
    {
        '$project': {
            'road': {
                '$map': {
                    'input': '$OpenDRIVE.road.planView.geometry', 
                    'as': 'road_array', 
                    'in': '$$road_array'
                }
            }, 
            'simulation_data': '$execution_data.simulation_data', 
            'meta_data': {
                'test_info': '$OpenDRIVE.header.sdc_test_info', 
                'run_config': '$OpenDRIVE.header.run_config'
            }
        }
    }, {
        '$project': {
            'road_points': {
                '$map': {
                    'input': '$road', 
                    'as': 'road', 
                    'in': {
                        'x': {
                            '$toDouble': '$$road.@x'
                        }, 
                        'y': {
                            '$toDouble': '$$road.@y'
                        }
                    }
                }
            }, 
            'trajectory': {
                '$map': {
                    'input': '$simulation_data', 
                    'as': 'data', 
                    'in': {
                        'time': '$$data.time', 
                        'x': {
                            '$arrayElemAt': [
                                '$$data.position', 0
                            ]
                        }, 
                        'y': {
                            '$arrayElemAt': [
                                '$$data.position', 1
                            ]
                        }
                    }
                }
            }, 
            'meta_data': {
                'test_info': {
                    'test_id': '$meta_data.test_info.@test_id', 
                    'test_outcome': '$meta_data.test_info.@test_outcome', 
                    'test_duration': {
                        '$toDouble': '$meta_data.test_info.@test_duration'
                    }, 
                    'is_valid': {
                        '$toBool': '$meta_data.test_info.@is_valid'
                    }
                }, 
                'run_config': {
                    'rf': {
                        '$toDouble': '$meta_data.run_config.@rf'
                    }, 
                    'oob': {
                        '$toDouble': '$meta_data.run_config.@oob'
                    }, 
                    'max_speed': {
                        '$toInt': '$meta_data.run_config.@max_speed'
                    }
                }
            }
        }
    }
]

if __name__ == "__main__":
    cli()



