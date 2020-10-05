from setuptools import setup, Extension

setup(
    name="cpp_disc_span_parser",
    version="0.1",
    include_dirs=["./cpp/"],
    ext_modules=[
        Extension(
            "cpp_disc_span_parser",
            sources=[
                "cpp/cpp_disc_span_parser.cpp",
                "cpp/disc_pstruct/binarize.cpp",
                "cpp/disc_pstruct/corpus.cpp",
                "cpp/disc_pstruct/set.cpp",
                "cpp/disc_pstruct/tree.cpp",
                "cpp/disc_pstruct/argmax-disc/chart.cpp",
                "cpp/disc_pstruct/inside-outside-disc/chart.cpp",
                "cpp/disc_pstruct/data.cpp",
		"cpp/disc_pstruct/argmax-disc/algorithms.cpp"
            ],
            language='c++',
            #extra_link_args=["-fopenmp"],
            extra_compile_args=[
                '-std=c++11',
                '-Wfatal-errors',
                '-Wall',
                '-Wextra',
                '-pedantic',
                '-O3',
                '-funroll-loops',
                '-march=native',
                '-fPIC',
                #'-fopenmp'
            ],
        )
    ]
);

