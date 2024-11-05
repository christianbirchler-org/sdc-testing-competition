proto:
	python -m grpc_tools.protoc \
		--proto_path=pb=./ 	\
		--python_out=./tools/clusterer 	\
		--pyi_out=./tools/clusterer	\
		--grpc_python_out=./tools/clusterer competition.proto

.PHONY: proto
