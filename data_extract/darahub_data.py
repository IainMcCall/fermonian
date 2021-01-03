import os
import pandas as pd
from datapackage import Package

# US GDP data
package = Package('https://datahub.io/core/gdp-us/datapackage.json')
print(package.resource_names)
for resource in package.resources:
    if resource.descriptor['datahub']['type'] == 'derived/csv':
        data = resource.read()
        for row in data:
            print(row)