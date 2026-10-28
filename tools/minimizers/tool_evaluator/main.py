#!python

import grpc
import pymongo

import interface_2027_pb2 as interface
import interface_2027_pb2_grpc as server_client

import click

# service CompetitionTool {
#   rpc Name(Empty) returns (NameReply) {}
# 
#   rpc Initialize (stream SDCTestExecution) returns (InitializationReply) {}
# 
#   rpc Minimize (stream SDCTestCase) returns (stream MinimizationReply) {}
# }

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
    te: interface.SDCTestExecution = None
    click.echo("initializing")
    click.echo("sensodat_uri: {}".format(sensodat_uri))
    mongo_client = pymongo.MongoClient(host=sensodat_uri)
    sensodat = mongo_client.get_database('sensodat')

    collection = sensodat.get_collection('campaign_11_frenetic')

    test_executions = []
    for doc in collection.aggregate(pipeline=SENSODAT_QUERY_PIPELINE):
        sdc_test_execution = _get_SDCTestExecution(doc)
        test_executions.append(sdc_test_execution)

    click.echo('nr of test executions: {}'.format(len(test_executions)))


def _get_SDCTestExecution(doc: dict) -> interface.SDCTestExecution:
    road_points= [interface.RoadPoint(
        sequenceNumber=ind,
        x=point['x'],
        y=point['y'],
    ) for ind, point in enumerate(doc['road_points'])]
    
    test_case = interface.SDCTestCase(testId=doc['meta_data']['test_info']['test_id'], roadPoints=road_points)

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


@cli.command(help="returns the index of the closest roadpoint to the OOB. Returns -1 on error.")
@click.argument("test-case", type=list[tuple[float, float]], help="test case: list of trajectory points")
def minimize(test_case: list[tuple[float, float]]):
    click.echo("minimizing")
    click.echo("testcases: {}".format(test_case))
    result = -1
    click.echo(result)

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



