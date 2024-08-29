# Competition Overview
The competition is in the context of regression testing for test suites executing system tests of self-driving cars in simulation.
Regression testing involves test selection, prioritization, and minimization [^3].
For the competition, we focus on the former aspect - the test selection.

## Test Selection
*Test Selection* is the process of picking only the relevant test cases from the test suite for a particular change.
In the context of simulation-based testing for SDCs with long running test cases, we select test cases fulfilling certain constraints [^1][^2].
For the competition we set a time budget:
- *Time Budget*: A maximal amount of time is available to run the selected test cases.

```{text}
 Test Suite        Selection
[ ][ ][ ][ ]      [x][ ][x][ ]
[ ][ ][ ][ ]  ->  [x][ ][ ][x]  -> Execution -> Results
[ ][ ][ ][ ]      [ ][x][ ][ ]
```
## Goal
The participants of the tool competition submit a test selector for simulation-based tests.
Specifically, the participants implement the predefined interfaces provide by the tool competition platform.

- *Time Budget*: A maximal amount of time is available to run the selected test cases.
- *Fault Detection*: TODO
- *Diversity*: TODO

## Competition Platform
The competition platform aims to provide the participant as much freedom as possible for their implementations.
The competitors can use any programming language they want.

To make the evaluation of the tools coherent, the competitors have to implement a gRPC [^4] interface specified in the `competition.proto` file.
The gRPC framework is language independent, i.e., there are various languages supported by gRPC.
The evaluator of the tools will invoke Remote Procedure Calls (RPC) which provide the tools the data for the evaluation.
Bewlow you see figures illustrating the overall set up.

```mermaid
block-beta
  columns 3
  d("Evaluator (Docker Container)"):1
  blockArrowId5<["gRPC"]>(x)
  g("ToolX (Docker Container)"):1
  block:group3:3
    docker("Docker with Nvidia Runtime")
  end

```

```mermaid
sequenceDiagram
    Evaluator ->>+ ToolX: initialize
    ToolX -->>- Evaluator: null
    Evaluator ->>+ ToolX: select
    ToolX -->>- Evaluator: return selection
```

## Competition Guidelines
There are no major limitations for the implementation of a test selection approach.
The competitors have only to implement the provided interfaces (`competition.proto`) and ensure that their tool works inside a Docker container.

The participants have to generate first the interface stubs using the `protoc` compile to generate the code based on the interface specification in the `competition.proto` file.

In `tools/sample_tool` is a sample implementation of a trivial test selector.
It starts a gRPC server and provides implementations of the interfaces.
Furthermore, the sample tool also provides a `Dockerfile` to run it inside a container.

Build and run the sample test selector tool:
```bash
cd tools/sample_tool
docker build -t my-selector-image .
docker run --rm --name my-selector-container -t -p 4545:4545 my-selector-image -p 4545
```

Build and run the evaluator tool that uses the sample dataset and performs an initial evaluation:
```bash
cd evaluator
docker build -t evaluator-image .
docker run --rm --name evaluator-container -t evaluator-image -u host.docker.internal:4545
```

All participants should ensure that the evaluator is able to provide an evaluation report to the console.
This is a clear indication the the gRPC interfaces are working properly, which is crucual for the competition's evaluation.


More information about gRPC you can find here: https://grpc.io/

## Evaluation
The organizers will evaluate the submitted tools on a virtual machine with the following specifications:

| **HW/SW** | **Requirement** |
|-----------|-----------------|
| CPU       | 8 vCPUs         |
| GPU       | Nvidia Tesla T4 |
| CUDA      | TBA             |
| RAM       | 8 GB            |
| OS        | Ubuntu/Linux    |
| Network   | no Internet     |


## Tool Submission
There are two ways to submit a tool:

### Open Source (preferred)
We ask the competitors to submit their tool by opening a Pull Request to this reposotiry.
The tool, i.e., the implementation of the provided interfaces, should be in the ´tool´ directory.
Furthermore, the competitors shall include a `LICENSE.md` for their tools.

The competition chairs will evaluate the submitted tools and in case of issues a discussion will happen in the Pull Request.

### Closed Source
In case of confidentiality reasons where the source code of the tool can not be disclosed, the competitors must submit their tool (copy of their repository) per email:

```text
TO: birc@zhaw.ch
SUBJECT: [ICST'25 SDC Tool Competition] Submission <TOOL NAME>
```


## References
[^1]: C. Birchler, S. Khatiri, B. Bosshard, A. Gambi, S. Panichella, "Machine learning-based test selection for simulation-based testing of self-driving cars software," Empirical Software Engineering (EMSE) 28, 71 (2023). https://doi.org/10.1007/s10664-023-10286-y
[^2]: C. Birchler, N. Ganz, S. Khatiri, A. Gambi and S. Panichella, "Cost-effective Simulation-based Test Selection in Self-driving Cars Software with SDC-Scissor," International Conference on Software Analysis, Evolution and Reengineering (SANER), 2022. https://doi.org/10.1109/SANER53432.2022.00030.
[^3]: Yoo, Shin, and Mark Harman. "Regression testing minimization, selection and prioritization: a survey." Software testing, verification and reliability 22.2 (2012): 67-120.
[^4]: https://grpc.io/
