#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <grpcpp/grpcpp.h>
#include "competition_2026.grpc.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerReaderWriter;
using grpc::Status;

struct TestFeatures {
    std::string test_id;
    double road_length;
    double curvature;
    double max_angle;
    int sharp_turns;
    double straightness;
    double complexity;
    double risk_score;
};

class MLPrioritizer final : public CompetitionTool::Service {
private:
    std::vector<TestFeatures> failed_test_features;
    
    // k parameter for k-NN algorithm
    const int K = 5;
    
    TestFeatures extract(const SDCTestCase& test) {
        TestFeatures features;
        features.test_id = test.testid();
        
        const auto& points = test.roadpoints();
        features.road_length = 0;
        features.curvature = 0;
        features.max_angle = 0;
        features.sharp_turns = 0;
        
        for (int i = 1; i < points.size(); i++) {
            double dx = points[i].x() - points[i-1].x();
            double dy = points[i].y() - points[i-1].y();
            features.road_length += std::sqrt(dx*dx + dy*dy);
            
            if (i >= 2) {
                double dx_prev = points[i-1].x() - points[i-2].x();
                double dy_prev = points[i-1].y() - points[i-2].y();
                double angle_prev = std::atan2(dy_prev, dx_prev);
                double angle_curr = std::atan2(dy, dx);
                double angle_diff = std::abs(angle_curr - angle_prev);
                if (angle_diff > M_PI) angle_diff = 2*M_PI - angle_diff;
                
                features.curvature += angle_diff;
                features.max_angle = std::max(features.max_angle, angle_diff);
                if (angle_diff > M_PI/6) features.sharp_turns++;
            }
        }
        
        double dx_total = points[points.size()-1].x() - points[0].x();
        double dy_total = points[points.size()-1].y() - points[0].y();
        double direct_distance = std::sqrt(dx_total*dx_total + dy_total*dy_total);
        features.straightness = direct_distance / (features.road_length + 1e-6);
        features.complexity = features.road_length * features.curvature;
        
        return features;
    }
    
    // Calculate distance between two test cases in feature space
    double calculateDistance(const TestFeatures& a, const TestFeatures& b) {
        // Normalize by expected ranges to make features comparable
        double d_length = (a.road_length - b.road_length) / 200.0;
        double d_curvature = (a.curvature - b.curvature) / 8.0;
        double d_complexity = (a.complexity - b.complexity) / 1500.0;
        double d_sharp = (a.sharp_turns - b.sharp_turns) / 10.0;
        
        // Euclidean distance in normalized feature space
        return std::sqrt(d_length * d_length + 
                        d_curvature * d_curvature + 
                        d_complexity * d_complexity +
                        d_sharp * d_sharp);
    }
    
    // ============================================
    // k-NN ALGORITHM IMPLEMENTATION
    // ============================================
    double calculateRiskUsingKNN(const TestFeatures& test) {
        if (failed_test_features.empty()) {
            return test.complexity;
        }
        
        // Step 1: Calculate distances to ALL training examples (failed tests)
        std::vector<std::pair<double, int>> distances;
        for (size_t i = 0; i < failed_test_features.size(); i++) {
            double dist = calculateDistance(test, failed_test_features[i]);
            distances.push_back({dist, i});
        }
        
        // Step 2: Sort by distance (ascending - closest first)
        std::sort(distances.begin(), distances.end(),
                 [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                     return a.first < b.first;
                 });
        
        // Step 3: Take only k nearest neighbors
        int k = std::min(K, (int)distances.size());
        
        // Step 4: Calculate weighted vote from k nearest neighbors
        // Closer neighbors have more influence (inverse distance weighting)
        double risk_score = 0.0;
        double total_weight = 0.0;
        
        for (int i = 0; i < k; i++) {
            double distance = distances[i].first;
            // Weight = 1 / (distance + small constant to avoid division by zero)
            double weight = 1.0 / (distance + 0.1);
            risk_score += weight;
            total_weight += weight;
        }
        
        // Normalize by total weight
        risk_score /= total_weight;
        
        return risk_score;
    }
    // ============================================
    
public:
    Status Name(ServerContext* context, const Empty* request, NameReply* reply) override {
        reply->set_name("cpp_knn_prioritizer");
        return Status::OK;
    }
    
    Status Initialize(ServerContext* context, ServerReader<Oracle>* reader, InitializationReply* reply) override {
        std::cout << "\n========================================" << std::endl;
        std::cout << "k-NN Test Prioritizer (k=" << K << ")" << std::endl;
        std::cout << "========================================" << std::endl;
        
        Oracle oracle;
        int total_tests = 0;
        int failed_count = 0;
        
        while (reader->Read(&oracle)) {
            if (oracle.hasfailed()) {
                failed_test_features.push_back(extract(oracle.testcase()));
                failed_count++;
            }
            total_tests++;
        }
        
        std::cout << "Training complete:" << std::endl;
        std::cout << "  Total tests: " << total_tests << std::endl;
        std::cout << "  Failed tests: " << failed_count << std::endl;
        std::cout << "  Using k=" << K << " nearest neighbors for prediction" << std::endl;
        std::cout << "========================================" << std::endl;
        
        reply->set_ok(true);
        return Status::OK;
    }
    
    Status Prioritize(ServerContext* context, 
                     ServerReaderWriter<PrioritizationReply, SDCTestCase>* stream) override {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Prioritizing with k-NN" << std::endl;
        std::cout << "========================================" << std::endl;
        
        std::vector<TestFeatures> tests;
        SDCTestCase test_case;
        while (stream->Read(&test_case)) {
            tests.push_back(extract(test_case));
        }
        
        std::cout << "Running k-NN on " << tests.size() << " tests..." << std::endl;
        
        // Apply k-NN algorithm to calculate risk score
        for (auto& test : tests) {
            test.risk_score = calculateRiskUsingKNN(test);
            
            // Apply boosting for high-risk characteristics
            if (test.complexity > 1000) test.risk_score *= 1.4;
            if (test.complexity > 1200) test.risk_score *= 1.3;
            if (test.road_length > 180) test.risk_score *= 1.2;
            if (test.curvature > 5) test.risk_score *= 1.2;
            if (test.straightness < 0.7) test.risk_score *= 1.15;
        }
        
        std::sort(tests.begin(), tests.end(),
            [](const TestFeatures& a, const TestFeatures& b) {
                return a.risk_score > b.risk_score;
            });
        
        std::cout << "k-NN prioritization complete" << std::endl;
        std::cout << "  Highest risk: " << tests[0].risk_score << std::endl;
        std::cout << "  Lowest risk: " << tests.back().risk_score << std::endl;
        std::cout << "========================================" << std::endl;
        
        for (const auto& test : tests) {
            PrioritizationReply reply;
            reply.set_testid(test.test_id);
            stream->Write(reply);
        }
        
        return Status::OK;
    }
};

int main(int argc, char** argv) {
    std::string port = "4545";
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-p" && i+1 < argc) {
            port = argv[++i];
        }
    }
    
    MLPrioritizer service;
    ServerBuilder builder;
    builder.AddListeningPort("0.0.0.0:" + port, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::unique_ptr<Server> server(builder.BuildAndStart());
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "k-NN Test Prioritizer" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Server on port " << port << std::endl;
    std::cout << "========================================" << std::endl;
    
    server->Wait();
    return 0;
}