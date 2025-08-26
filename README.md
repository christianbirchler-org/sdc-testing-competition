# SDC Testing Competition
![Static Badge](https://img.shields.io/badge/Python-3.11-blue)
![GitHub Discussions](https://img.shields.io/github/discussions/christianbirchler-org/sdc-testing-competition)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/christianbirchler-org/sdc-testing-competition)

This repository contains information and code for the tool competition on test selection for self-driving cars in simulation.

Self-driving cars (SDCs) are equipped with onboard cameras and various sensors have already demonstrated the possibility of autonomous driving in real environments, leading to great interest in various application domains.
However, despite the necessity of systematically testing such complex and automated systems to ensure their safe operation in real-world environments, there has been relatively limited investment in this direction so far.

The SDC Testing Competition is an initiative designed to inspire and encourage the Software Testing Community to direct their attention toward SDCs as a rapidly emerging and crucial domain.
The competition's current focus is on regression testing for test suites executing system tests of SDCs in simulation.
Regression testing involves test selection, prioritization, and minimization.
![](example.png)

## Quickstart

### Taking part in a competition
Clone the repository and fetch large files from Git LFS.
``` bash
git clone git@github.com:christianbirchler-org/sdc-testing-competition.git
git lfs fetch
```

Read the competition instruction:
``` bash
cat competition/<YEAR>.md # e.g., cat competition/2026.md
```

Generate stubs for the interfaces, that need an implementation for the competition.
On the gRPC website is a list of [supported languages](https://grpc.io/docs/languages/) and instructions to generate stubs for the interfaces defined in the `.proto` files.

### Evaluate the tools
We provide for each competition an evaluation tool that acts as a gRPC client.
First, start your tool, which is a gRPC service.
Secondly, run the evaluation tool.
The output of the evaluation tool should provide a summary of the evaluation metrics.

## Q&A
Use [GitHub Discussions](https://github.com/christianbirchler-org/sdc-testing-competition/discussions) for any kind of questions related to the tool competition.

> Do not hesitate to ask questions.
> If something is unclear then it is likely it is unclear for others as well.
> We appreciate all kind of feedback to make this competition platform as usable as possible.


## Contributing
The important dates for the competition and the current development of this competition platform are defined as [GitHub Milestones](https://github.com/christianbirchler-org/sdc-testing-competition/milestones).

Any kind of contributions (e.g., feature requests, bug reports, questions, etc.) are welcome.
Please refer to [GitHub Discussions](https://github.com/christianbirchler-org/sdc-testing-competition/discussions) to let the community know about your contributions.

### Contributors
Many thanks to the following contributors who help to build and maintain the repository:
- [ChristianBirchler](https://github.com/ChristianBirchler)
- [vatozZ](https://github.com/vatozZ)
- [FasihMunirMalik](https://github.com/FasihMunirMalik)

## License
```{text}
SDC Testing Competition Platform
Copyright (C) 2025  Christian Birchler

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```
[GPLv3](LICENSE)

## Contacts
- Responsible of the repository: [Christian Birchler](https://www.christianbirchler.org)
- [Competition Organizers](./competitions/README.md)
