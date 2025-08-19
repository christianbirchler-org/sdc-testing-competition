# Random Test Selector
This is a sample implementation of a random test selector for the SDC Tool Competition.

## Use
```bash
cd tools/sample_tool
docker build -t my-selector-image .
docker run --rm --name my-selector-container -t -p 4545:4545 my-selector-image -p 4545
```
## License
```{text}
Random Test Selection Tool
Copyright (C) 2024  Christian Birchler

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

