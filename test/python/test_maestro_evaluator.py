from domino import utils
import os


def test_find_path():
    maestro_path = utils.find_maestro()
    print(maestro_path)


def test_example_mapping_nvdla():
    # MaestroNVDLA-stype dataflow
    mapping_contents = """
    Network sample_net {
    Layer Conv2d-1 {
    Type: CONV
    Stride { X: 1, Y: 1 }
    Dimensions { K: 512, C: 512, R: 3, S: 3, Y: 7, X: 7 }
    Dataflow {
            SpatialMap(1,1) K;
            TemporalMap(64,64) C;
            TemporalMap(Sz(R),Sz(R)) R;
            TemporalMap(Sz(S),Sz(S)) S;
            TemporalMap(Sz(R),1) Y;
            TemporalMap(Sz(S),1) X;
            Cluster(64, P);
            SpatialMap(1,1) C;
            TemporalMap(Sz(R),1) Y;
            TemporalMap(Sz(S),1) X;
            TemporalMap(Sz(R),Sz(R)) R;
            TemporalMap(Sz(S),Sz(S)) S;
    }
    }
    }
    """

    mapping_contents = """
    Network sample_net {
    Layer Bottleneck6_1_2 {
		Type: DSCONV
		Stride { X: 2, Y: 2 }		
		Dimensions { K: 1, C: 576, R: 3, S: 3, Y: 14, X: 14 }
		Dataflow {
			TemporalMap(1,1) C;
			SpatialMap(Sz(R), 1) Y;
			TemporalMap(10,8) X;
			TemporalMap(Sz(R), Sz(R)) R;
			TemporalMap(Sz(S), Sz(S)) S;						
			Cluster(8, P);
			SpatialMap(Sz(S), 1) X;
		}				
	}
    }
    """

    mapping_contents = """
    Network BLAS3 {
	Layer BLAS {
		Type: GEMM
		Dimensions { K: 100, M: 100, N: 100 }
		Dataflow {
			SpatialMap(32, 32) M;
            SpatialMap(32, 32) N;
            TemporalMap(32, 32) K;
            TemporalMap(16, 16) M;
            Cluster(32, P);
            SpatialMap(16, 16) N;
            SpatialMap(16, 16) K;
		}
	}
}
    """
    mapping_file = "sample_mapping"

    if os.path.exists(f"{mapping_file}.m") and os.path.isfile(f"{mapping_file}.m"):
        os.remove(f"{mapping_file}.m")

    with open(f"{mapping_file}.m", "w") as fout:
        fout.write(mapping_contents)

    maestro_path = utils.find_maestro()
    command = utils.generate_maestro_command(
        maestro_path,
        mapping_file,
        1000,  # noc_bw,
        50,  # off_chip_bw,
        256,  # num_pes,
        100,  # l1_size,
        3000,  # l2_size,
    )

    results = utils.run_maestro(mapping_file, command)

    if os.path.exists(f"{mapping_file}.m") and os.path.isfile(f"{mapping_file}.m"):
        os.remove(f"{mapping_file}.m")

    print(results)


def test_example_mapping_shidiannao():
    # ShiDianNao-stype dataflow
    mapping_contents = """
    Network sample_net {
    Layer CONV1 {
		Type: CONV
		Stride { X: 2, Y: 2 }		
		Dimensions { K: 64, C: 3, R: 7, S: 7, Y:224, X:224 }
		Dataflow {
			TemporalMap(1,1) K;
			TemporalMap(1,1) C;
			SpatialMap(Sz(R), 1) Y;
			TemporalMap(8,8) X;
			TemporalMap(Sz(R), Sz(R)) R;
			TemporalMap(Sz(S), Sz(S)) S;
			Cluster(8, P);
			SpatialMap(Sz(S), 1) X;
		}
	}
    }
    """
    mapping_file = "sample_mapping"

    if os.path.exists(f"{mapping_file}.m") and os.path.isfile(f"{mapping_file}.m"):
        os.remove(f"{mapping_file}.m")

    with open(f"{mapping_file}.m", "w") as fout:
        fout.write(mapping_contents)

    maestro_path = utils.find_maestro()
    command = utils.generate_maestro_command(
        maestro_path,
        mapping_file,
        1000,  # noc_bw,
        50,  # off_chip_bw,
        256,  # num_pes,
        100,  # l1_size,
        3000,  # l2_size,
    )

    results = utils.run_maestro(mapping_file, command)

    if os.path.exists(f"{mapping_file}.m") and os.path.isfile(f"{mapping_file}.m"):
        os.remove(f"{mapping_file}.m")

    print(results)


def test_example_mapping_eyeriss():
    # Eyeriss-stype dataflow
    mapping_contents = """
    Network sample_net {
    Layer Conv2d-1 {
    Type: CONV
    Stride { X: 2, Y: 2 }
    Dimensions { K: 64, C: 3, R: 7, S: 7, Y: 224, X: 224 }
    Dataflow {
        SpatialMap(1,1) Y';
        TemporalMap(1,1) X';
        TemporalMap(1,1) C;
        TemporalMap(16,16) K;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
        Cluster(Sz(R),P);
        SpatialMap(1,1) Y;
        SpatialMap(1,1) R;
        TemporalMap(Sz(S),Sz(S)) S;
    }
    }
    }
    """
    mapping_file = "sample_mapping"

    if os.path.exists(f"{mapping_file}.m") and os.path.isfile(f"{mapping_file}.m"):
        os.remove(f"{mapping_file}.m")

    with open(f"{mapping_file}.m", "w") as fout:
        fout.write(mapping_contents)

    maestro_path = utils.find_maestro()
    command = utils.generate_maestro_command(
        maestro_path,
        mapping_file,
        1000,  # noc_bw,
        50,  # off_chip_bw,
        256,  # num_pes,
        100,  # l1_size,
        3000,  # l2_size,
    )

    results = utils.run_maestro(mapping_file, command)

    if os.path.exists(f"{mapping_file}.m") and os.path.isfile(f"{mapping_file}.m"):
        os.remove(f"{mapping_file}.m")

    print(results)


def test_example_mapping_tpu():
    # TPU-stype dataflow
    mapping_contents = """
    Network sample_net {
    Layer Conv2d-1 {
    Type: CONV
    Stride { X: 2, Y: 2 }
    Dimensions { K: 64, C: 3, R: 7, S: 7, Y: 224, X: 224 }
    Dataflow {
        TemporalMap(16,16) K;
        SpatialMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        TemporalMap(1,1) C;
        Cluster(16, P);
        SpatialMap(1,1) K;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        TemporalMap(Sz(R),7) R;
        TemporalMap(Sz(S),7) S;
    }
    }
    }
    """
    mapping_file = "sample_mapping"

    if os.path.exists(f"{mapping_file}.m") and os.path.isfile(f"{mapping_file}.m"):
        os.remove(f"{mapping_file}.m")

    with open(f"{mapping_file}.m", "w") as fout:
        fout.write(mapping_contents)

    maestro_path = utils.find_maestro()
    command = utils.generate_maestro_command(
        maestro_path,
        mapping_file,
        1000,  # noc_bw,
        50,  # off_chip_bw,
        256,  # num_pes,
        100,  # l1_size,
        3000,  # l2_size,
    )

    results = utils.run_maestro(mapping_file, command)

    if os.path.exists(f"{mapping_file}.m") and os.path.isfile(f"{mapping_file}.m"):
        os.remove(f"{mapping_file}.m")

    print(results)


if __name__ == "__main__":
    test_find_path()
    test_example_mapping_nvdla()
    test_example_mapping_shidiannao()
    test_example_mapping_eyeriss()
    test_example_mapping_tpu()
