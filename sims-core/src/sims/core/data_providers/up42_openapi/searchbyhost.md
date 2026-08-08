---
updatedAt: 2026-05-27T08:26:46.000Z
---

Fetch the complete documentation index at: https://developer.up42.com/llms.txt. Use this file to discover all available pages before exploring further.

# Search the catalog by host name

Get a list of available full scenes from the catalog.


# OpenAPI definition

```json
{
  "openapi": "3.0.0",
  "info": {
    "title": "UP42 API",
    "contact": {
      "name": "Contact support",
      "email": "support@up42.com"
    },
    "license": {
      "name": "Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License",
      "url": "http://creativecommons.org/licenses/by-nc-nd/4.0/"
    },
    "version": "1.2"
  },
  "servers": [
    {
      "url": "https://api.up42.com"
    }
  ],
  "security": [
    {
      "httpBearer": []
    }
  ],
  "tags": [
    {
      "name": "Catalog"
    }
  ],
  "paths": {
    "/catalog/hosts/{host-name}/stac/search": {
      "post": {
        "tags": [
          "Catalog"
        ],
        "summary": "Search the catalog by host name",
        "description": "Get a list of available full scenes from the catalog.\n",
        "operationId": "searchByHost",
        "parameters": [
          {
            "$ref": "#/components/parameters/HostNamePath"
          },
          {
            "name": "next",
            "in": "query",
            "description": "Search for previous (`prev:{ID}`) or next (`next:{ID}`) full scenes from the results page.\n",
            "required": false,
            "style": "form",
            "explode": true,
            "schema": {
              "type": "string",
              "example": "next:1b4d92d4-dd58-484a-9ebb-11c668d58776"
            }
          }
        ],
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/SearchRequestDto"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "OK",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/StacResponse"
                }
              }
            }
          },
          "401": {
            "$ref": "#/components/responses/CatalogUnauthorized"
          },
          "404": {
            "$ref": "#/components/responses/ResourceNotFound"
          },
          "422": {
            "$ref": "#/components/responses/GeometryError"
          },
          "542": {
            "$ref": "#/components/responses/HostError"
          }
        }
      }
    }
  },
  "components": {
    "schemas": {
      "Link": {
        "type": "object",
        "properties": {
          "rel": {
            "type": "string",
            "example": "self"
          },
          "href": {
            "type": "string",
            "example": "https://api.up42.dev/catalog/hosts/oneatlas/stac/search"
          }
        }
      },
      "SearchRequestDto": {
        "type": "object",
        "properties": {
          "collections": {
            "uniqueItems": true,
            "type": "array",
            "description": "The names of the collections whose full scenes you want to include in search results.\n",
            "nullable": true,
            "example": [
              "pneo",
              "phr"
            ],
            "items": {
              "type": "string"
            }
          },
          "ids": {
            "uniqueItems": true,
            "type": "array",
            "description": "The IDs of specific full scenes you want to include in search results.\n",
            "nullable": true,
            "example": [
              "f6b86de1-a9ea-4322-99fe-23126d2784b6"
            ],
            "items": {
              "type": "string"
            }
          },
          "datetime": {
            "type": "string",
            "description": "Search for full scenes that have a temporal property that intersects the datetime value in the RFC 3339 format. You can search for a specific date and time, or for a closed or an open date interval. Express open intervals using double-dots.\n\n- A timestamp: `2025-02-12T23:20:50Z`\n- A closed interval: `2025-02-12T00:00:00Z/2025-03-18T12:31:12Z`\n- An interval without end date: `2025-02-12T00:00:00Z/..`\n- An interval without start date: `../2025-03-18T12:31:12Z`\n",
            "nullable": true,
            "example": "2025-01-01T00:00:00Z/2025-01-15T23:59:59Z"
          },
          "limit": {
            "maximum": 500,
            "minimum": 1,
            "type": "integer",
            "description": "The number of full scenes on a results page.\n",
            "format": "int32",
            "nullable": true,
            "example": 100
          },
          "query": {
            "$ref": "#/components/schemas/CatalogStacQuery"
          },
          "bbox": {
            "maxItems": 4,
            "minItems": 4,
            "type": "array",
            "description": "A search geometry in the GeoJSON format. Returns images that intersect with the defined rectangle and may not fully cover it. Use only if `intersects` isn't specified.\n",
            "nullable": true,
            "example": [
              6.553674,
              62.191632,
              6.560222,
              62.195116
            ],
            "items": {
              "type": "number",
              "format": "double"
            }
          },
          "intersects": {
            "$ref": "#/components/schemas/Polygon"
          }
        }
      },
      "Properties": {
        "required": [
          "collection",
          "constellation",
          "id",
          "producer",
          "providerName",
          "providerProperties",
          "sceneId",
          "up42:usageType"
        ],
        "type": "object",
        "properties": {
          "id": {
            "type": "string",
            "description": "The full scene ID. Use for data ordering.\n",
            "example": "5ad07b37-f6a1-4829-bc74-0fe50528b0ed"
          },
          "acquisitionDate": {
            "type": "string",
            "description": "The date and time when the sensor acquired the data.\n",
            "format": "date-time",
            "nullable": true,
            "example": "2019-03-23T10:24:03.556Z",
            "deprecated": true
          },
          "datetime": {
            "type": "string",
            "description": "The date and time when the sensor acquired the data.\n",
            "format": "date-time",
            "nullable": true,
            "example": "2019-03-24T12:12:00.556Z"
          },
          "start_datetime": {
            "type": "string",
            "description": "The date and time when the sensor started the acquisition process.\n",
            "format": "date-time",
            "nullable": true,
            "example": "2019-03-23T10:24:03.556Z"
          },
          "end_datetime": {
            "type": "string",
            "description": "The date and time when the sensor finished the acquisition process.\n",
            "format": "date-time",
            "nullable": true,
            "example": "2019-03-25T11:11:00.556Z"
          },
          "constellation": {
            "type": "string",
            "description": "The name of the sensor.\n",
            "example": "pneo"
          },
          "collection": {
            "type": "string",
            "description": "The name of the collection.\n",
            "example": "pneo"
          },
          "providerName": {
            "type": "string",
            "example": "oneatlas",
            "deprecated": true
          },
          "cloudCoverage": {
            "type": "number",
            "description": "The percentage of cloud coverage.\n",
            "format": "double",
            "nullable": true,
            "example": 0
          },
          "up42:usageType": {
            "uniqueItems": true,
            "type": "array",
            "description": "The type of usage.\n",
            "deprecated": true,
            "items": {
              "type": "string",
              "enum": [
                "DATA",
                "ANALYTICS"
              ]
            }
          },
          "providerProperties": {
            "type": "object",
            "additionalProperties": true,
            "deprecated": true
          },
          "sceneId": {
            "type": "string",
            "description": "The additional full scene ID. Don't use for data ordering.\n",
            "example": "DS_PHR1B_201903231024035_FR1_PX_E013N52_0915_02862"
          },
          "resolution": {
            "type": "number",
            "description": "The spatial resolution, in meters.\n",
            "format": "double",
            "example": 0.5
          },
          "deliveryTime": {
            "type": "string",
            "description": "The unit of data delivery time.\n",
            "example": "MINUTES",
            "enum": [
              "MINUTES",
              "HOURS",
              "DAYS"
            ]
          },
          "producer": {
            "type": "string",
            "description": "The name of the producer.\n\nData producers are companies that initially acquired and processed the source data. Data acquired by a producer can be distributed to various hosts.\n",
            "example": "Airbus"
          }
        }
      },
      "StacResponse": {
        "required": [
          "features",
          "links",
          "type"
        ],
        "type": "object",
        "properties": {
          "features": {
            "type": "array",
            "description": "A list of full scenes with the defined parameters from the chosen host.\n",
            "items": {
              "$ref": "#/components/schemas/CatalogFeature"
            }
          },
          "links": {
            "type": "array",
            "description": "A list of links related to the current endpoint. Use to navigate through objects.\n",
            "items": {
              "$ref": "#/components/schemas/Link"
            }
          },
          "type": {
            "type": "string",
            "description": "The type of GeoJSON object.\n",
            "example": "FeatureCollection",
            "default": "FeatureCollection"
          }
        }
      },
      "Polygon": {
        "required": [
          "coordinates",
          "type"
        ],
        "type": "object",
        "properties": {
          "type": {
            "type": "string",
            "description": "The type of geometry. The only allowed value is `Polygon`.",
            "example": "Polygon",
            "enum": [
              "Polygon"
            ]
          },
          "coordinates": {
            "type": "array",
            "description": "The coordinates.",
            "example": [
              [
                [
                  13.37684864751185,
                  52.526224085531766
                ],
                [
                  13.37684864751185,
                  52.497575021739664
                ],
                [
                  13.423387693820644,
                  52.497575021739664
                ],
                [
                  13.423387693820644,
                  52.526224085531766
                ],
                [
                  13.37684864751185,
                  52.526224085531766
                ]
              ]
            ],
            "items": {
              "type": "array",
              "items": {
                "type": "array",
                "items": {
                  "type": "number",
                  "format": "double"
                }
              }
            }
          }
        },
        "description": "A polygon in the GeoJSON format.\n",
        "nullable": true,
        "externalDocs": {
          "url": "http://geojson.org/geojson-spec.html#polygon"
        }
      },
      "CatalogFeature": {
        "required": [
          "geometry",
          "properties",
          "type"
        ],
        "type": "object",
        "properties": {
          "geometry": {
            "$ref": "#/components/schemas/CatalogGeometry"
          },
          "properties": {
            "$ref": "#/components/schemas/Properties"
          },
          "bbox": {
            "type": "array",
            "example": [
              6.553674,
              62.191632,
              6.560222,
              62.195116
            ],
            "items": {
              "type": "number"
            }
          },
          "type": {
            "type": "string",
            "default": "Feature"
          }
        },
        "description": "A list of full scenes with the defined parameters from the chosen host.\n"
      },
      "CatalogStacQuery": {
        "type": "object",
        "properties": {
          "cloudCoverage": {
            "maxProperties": 1,
            "minProperties": 1,
            "type": "object",
            "properties": {
              "GT": {
                "type": "integer",
                "description": "Greater than"
              },
              "GTE": {
                "type": "integer",
                "description": "Greater than or equal to"
              },
              "LT": {
                "type": "integer",
                "description": "Less than"
              },
              "LTE": {
                "type": "integer",
                "description": "Less than or equal to"
              }
            },
            "additionalProperties": false,
            "description": "Search by the percentage of cloud coverage. The format is `<operator>: <integer>`. Replace `<operator>` with a comparison operator:\n\n- `GT`: greater than\n- `GTE`: greater than or equal to\n- `LT`: less than\n- `LTE`: less than or equal to\n",
            "example": {
              "LT": 10
            }
          }
        },
        "description": "A STAC query object.\n",
        "nullable": true
      },
      "CatalogGeometry": {
        "required": [
          "coordinates",
          "type"
        ],
        "type": "object",
        "properties": {
          "type": {
            "type": "string",
            "description": "The type of geometry. Only AOIs are supported.",
            "example": "Polygon",
            "enum": [
              "Polygon"
            ]
          },
          "coordinates": {
            "type": "array",
            "description": "The coordinates.",
            "example": [
              [
                [
                  77.259584,
                  28.622331
                ],
                [
                  77.287879,
                  28.622632
                ],
                [
                  77.287621,
                  28.594794
                ],
                [
                  77.259389,
                  28.594983
                ],
                [
                  77.259584,
                  28.622331
                ]
              ]
            ],
            "items": {
              "type": "array",
              "items": {
                "type": "array",
                "items": {
                  "type": "number"
                }
              }
            }
          }
        },
        "description": "The geometry in the GeoJSON format.\n",
        "externalDocs": {
          "url": "http://geojson.org/geojson-spec.html#geometry-objects"
        }
      }
    },
    "responses": {
      "GeometryError": {
        "description": "Request not valid",
        "content": {
          "application/json": {
            "schema": {
              "type": "object",
              "properties": {
                "data": {
                  "type": "object",
                  "nullable": true,
                  "example": null
                },
                "error": {
                  "type": "object",
                  "properties": {
                    "code": {
                      "type": "integer",
                      "format": "int32",
                      "example": 422
                    },
                    "message": {
                      "type": "string",
                      "example": "1 validation error for Request\nbody -> intersects -> coordinates\nGeometry should not have more than 999 vertices. (type=value_error)\n"
                    },
                    "details": {
                      "type": "object",
                      "nullable": true,
                      "example": null
                    }
                  }
                }
              }
            }
          }
        }
      },
      "ResourceNotFound": {
        "description": "Resource not found",
        "content": {
          "application/json": {
            "schema": {
              "type": "object",
              "properties": {
                "data": {
                  "type": "string",
                  "format": "nullable"
                },
                "error": {
                  "type": "object",
                  "properties": {
                    "code": {
                      "type": "integer",
                      "format": "int32",
                      "example": 404
                    },
                    "message": {
                      "type": "string",
                      "example": "Could not find host 'pleiades'."
                    },
                    "details": {
                      "type": "string",
                      "format": "nullable"
                    }
                  }
                }
              }
            }
          }
        }
      },
      "CatalogUnauthorized": {
        "description": "Unauthorized",
        "content": {
          "application/json": {
            "schema": {
              "type": "object",
              "properties": {
                "data": {
                  "type": "object",
                  "nullable": true,
                  "example": null
                },
                "error": {
                  "type": "object",
                  "properties": {
                    "code": {
                      "type": "integer",
                      "example": 401
                    },
                    "message": {
                      "type": "string",
                      "example": "An error occurred while attempting to decode the Jwt: Jwt expired at 2023-02-15T11:29:44Z\n"
                    },
                    "details": {
                      "type": "object",
                      "nullable": true,
                      "example": null
                    }
                  }
                }
              }
            }
          }
        }
      },
      "HostError": {
        "description": "Request failed due to an outage on the host side",
        "content": {
          "application/json": {
            "schema": {
              "type": "object",
              "properties": {
                "data": {
                  "type": "object",
                  "nullable": true,
                  "example": null
                },
                "error": {
                  "type": "object",
                  "properties": {
                    "code": {
                      "type": "integer",
                      "format": "int32",
                      "example": 542
                    },
                    "message": {
                      "type": "string",
                      "example": "Operation failed."
                    },
                    "details": {
                      "type": "object",
                      "nullable": true,
                      "example": null
                    }
                  }
                }
              }
            }
          }
        }
      }
    },
    "parameters": {
      "HostNamePath": {
        "name": "host-name",
        "in": "path",
        "description": "The name of the data host.\n",
        "required": true,
        "style": "simple",
        "explode": false,
        "schema": {
          "type": "string",
          "example": "oneatlas"
        }
      }
    },
    "securitySchemes": {
      "httpBearer": {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT"
      }
    }
  }
}
```