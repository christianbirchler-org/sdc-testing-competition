#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <grpcpp/grpcpp.h>
#include "competition_2026.grpc.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerReaderWriter;
using grpc::Status;

class SamplePrioritizer final : public CompetitionTool::Service {
    Status Name(ServerContext* context, const Empty* request, NameReply* reply) override {
        reply->set_name("cpp_sample_prioritizer");
        std::cout << "[INFO] Name requested" << std::endl;
        return Status::OK;
    }
    
    Status Initialize(ServerContext* context, ServerReader<Oracle>* reader, InitializationReply* reply) override {
        std::cout << "[INFO] Initialization started" << std::endl;
        Oracle oracle;
        int count = 0;
        while (reader->Read(&oracle)) {
            count++;
        }
        std::cout << "[INFO] Received " << count << " training tests" << std::endl;
        reply->set_ok(true);
        return Status::OK;
    }
    
    Status Prioritize(ServerContext* context, ServerReaderWriter<PrioritizationReply, SDCTestCase>* stream) override {
        std::cout << "[INFO] Prioritization started" << std::endl;
        std::vector<SDCTestCase> tests;
        SDCTestCase test;
        
        // Read all tests
        while (stream->Read(&test)) {
            tests.push_back(test);
        }
        std::cout << "[INFO] Received " << tests.size() << " tests to prioritize" << std::endl;
        
        // Random shuffle
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(tests.begin(), tests.end(), g);
        
        // Write back prioritized order
        for (const auto& t : tests) {
            PrioritizationReply reply;
            reply.set_testid(t.testid());
            stream->Write(reply);
        }
        std::cout << "[INFO] Prioritization complete" << std::endl;
        return Status::OK;
    }
};

int main(int argc, char** argv) {
    std::string port = "4545";
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-p" || arg == "--port") && i + 1 < argc) {
            port = argv[++i];
        }
    }
    
    std::string address = "0.0.0.0:" + port;
    SamplePrioritizer service;
    ServerBuilder builder;
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << address << std::endl;
    server->Wait();
    return 0;
}