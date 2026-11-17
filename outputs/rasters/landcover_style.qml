<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28.0" styleCategories="AllStyleCategories">
  <pipe>
    <provider>
      <resampling enabled="false" maxOversampling="2"/>
    </provider>
    <rasterrenderer type="paletted" band="1" opacity="1">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>None</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Exact</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <colorPalette>
        <paletteEntry value="1" color="#1b7837" alpha="255" label="Bosque"/>
        <paletteEntry value="2" color="#a6d96a" alpha="255" label="Pastos y Cultivos"/>
        <paletteEntry value="3" color="#d73027" alpha="255" label="Suelo Desnudo y Roca"/>
        <paletteEntry value="4" color="#e08214" alpha="255" label="Urbano y Vías"/>
        <paletteEntry value="5" color="#2166ac" alpha="255" label="Agua"/>
      </colorPalette>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0"/>
    <huesaturation colorizeGreen="128" colorizeRed="255" colorizeBlue="128" grayscaleMode="0" saturation="0" colorizeOn="0" colorizeStrength="100"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
