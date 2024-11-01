proto:
	python -m grpc_tools.protoc \
		--proto_path=pb=./ 	\
		--python_out=./selector 	\
		--pyi_out=./selector	\
		--grpc_python_out=./selector competition.proto

.PHONY: proto
