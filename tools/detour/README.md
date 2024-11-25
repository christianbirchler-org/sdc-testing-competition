# DETOUR
DETOUR is a test selector for the SDC Tool Competition.

## Use
```bash
cd tools/detour
docker build -t detour-image .
docker run --rm --name detour-container -t -p 4545:4545 detour-image -p 4545
```

## Authors

Paolo Arcaini (National Institute of Informatics, Japan)
Ahmet Cetinkaya (Shibaura Institute of Technology, Japan)
