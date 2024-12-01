#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_generators.hpp>
#include <cugraph/legacy/functions.hpp>  // legacy coo_to_csr


#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <iostream>

#include "mmio.h"

void mtx_parser(const std::string& file_path, std::vector<int32_t>& h_src, std::vector<int32_t>& h_dst, std::vector<float>& h_weights) {
    FILE* f = fopen(file_path.c_str(), "r");
    if (f == NULL) {
        std::cerr << "Error opening file: " << file_path << std::endl;
        exit(1);
    }

    MM_typecode matcode;
    if (mm_read_banner(f, &matcode) != 0) {
        std::cerr << "Error reading matrix banner" << std::endl;
        exit(1);
    }

    if (!mm_is_matrix(matcode) || !mm_is_sparse(matcode) || !mm_is_real(matcode)) {
        std::cerr << "This example only works with real-valued sparse matrices" << std::endl;
        exit(1);
    }

    int M, N, nz;
    if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0) {
        std::cerr << "Error reading matrix size" << std::endl;
        exit(1);
    }

    h_src.resize(nz);
    h_dst.resize(nz);
    h_weights.resize(nz);

    for (int i = 0; i < nz; i++) {
        if (fscanf(f, "%d %d %f\n", &h_src[i], &h_dst[i], &h_weights[i]) != 3) {
            std::cerr << "Error reading matrix entry" << std::endl;
            exit(1);
        }
        h_src[i]--;
        h_dst[i]--;
    }
    fclose(f);
}


int main(int argc, char** argv) {
    // This is looks necessary to run cugraph algorithms?
    raft::handle_t handle{};

    /*
    // Create example graph data - a simple graph with 5 vertices and 6 edges
    std::vector<int32_t> h_src = {0, 1, 1, 2, 2, 3};
    std::vector<int32_t> h_dst = {1, 2, 3, 3, 4, 4};
    std::vector<float> h_weights = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    */

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path to mtx file>" << std::endl;
        exit(1);
    } else {
        std::cout << "Reading graph from file: " << argv[1] << std::endl;
    }

    std::string file_path = argv[1];

    std::vector<int32_t> h_src;
    std::vector<int32_t> h_dst;
    std::vector<float> h_weights;

    mtx_parser(file_path, h_src, h_dst, h_weights);

    // Create device vectors
    rmm::device_uvector<int32_t> d_src(h_src.size(), handle.get_stream());
    rmm::device_uvector<int32_t> d_dst(h_dst.size(), handle.get_stream());
    rmm::device_uvector<float> d_weights(h_weights.size(), handle.get_stream());

    // Copy data to GPU
    cudaMemcpy(d_src.data(), h_src.data(), sizeof(int32_t) * h_src.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst.data(), h_dst.data(), sizeof(int32_t) * h_dst.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights.data(), h_weights.data(), sizeof(float) * h_weights.size(), cudaMemcpyHostToDevice);

    // Create graph
    cugraph::graph_t<int32_t, int32_t, false, false> graph(handle);
    std::optional<cugraph::edge_property_t<cugraph::graph_view_t<int32_t, int32_t, false, false>, float>> edge_weights;
    std::optional<rmm::device_uvector<int32_t>> renumber_map{std::nullopt};
    std::tie(graph, edge_weights, std::ignore, std::ignore, renumber_map) = 
        cugraph::create_graph_from_edgelist<int32_t, int32_t, float, int32_t, int32_t, false, false>(
            handle,
            std::nullopt,              // No vertex list needed
            std::move(d_src),
            std::move(d_dst),
            std::make_optional(std::move(d_weights)),
            std::nullopt,              // No edge ids needed
            std::nullopt,              // No edge types needed
            cugraph::graph_properties_t{false, false},  // Not symmetric, allow multi-edges
            false                      // Don't renumber
        );

    // Prepare for Louvain ??? What is clustering?
    rmm::device_uvector<int32_t> clustering(5, handle.get_stream()); // Size = number of vertices

    // Run Louvain
    auto [num_levels, modularity] = cugraph::louvain(
        handle, 
        std::optional<std::reference_wrapper<raft::random::RngState>>{std::nullopt},
        graph.view(),
        edge_weights ? std::make_optional(edge_weights->view()) : std::nullopt,
        clustering.data(),
        100,    // max_level
        1e-7f,  // threshold
        1.0f    // resolution
    );

    // What is clustering?
    std::vector<int32_t> h_clustering(5);
    cudaMemcpy(h_clustering.data(), clustering.data(), sizeof(int32_t) * 5, cudaMemcpyDeviceToHost);

    std::cout << "Number of levels: " << num_levels << std::endl;
    std::cout << "Modularity: " << modularity << std::endl;
    std::cout << "Communities: ";
    for (auto c : h_clustering) {
        std::cout << c << " ";
    }
    std::cout << std::endl;

    return 0;
}