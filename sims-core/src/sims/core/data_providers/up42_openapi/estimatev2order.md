---
updatedAt: 2026-05-27T08:26:46.000Z
---

Fetch the complete documentation index at: https://developer.up42.com/llms.txt. Use this file to discover all available pages before exploring further.

# Estimate the cost of an order

Get a cost estimation before creating a tasking or a catalog order.

You can receive an overview of the overall credit amount that will be deducted from your credit balance if you decide to proceed with the ordering.


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
      "name": "Orders"
    }
  ],
  "paths": {
    "/v2/orders/estimate": {
      "post": {
        "tags": [
          "Orders"
        ],
        "summary": "Estimate the cost of an order",
        "description": "Get a cost estimation before creating a tasking or a catalog order.\n\nYou can receive an overview of the overall credit amount that will be deducted from your credit balance if you decide to proceed with the ordering.\n",
        "operationId": "estimateV2Order",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/CreateV2OrderRequest"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Cost estimation returned. Any geometry validation failures will be returned inside the `errors` array\n",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/EstimateV2OrderResponse"
                }
              }
            }
          },
          "400": {
            "$ref": "#/components/responses/BadRequest"
          },
          "401": {
            "$ref": "#/components/responses/OrderingUnauthorized"
          }
        }
      }
    }
  },
  "components": {
    "schemas": {
      "Tag": {
        "maxLength": 180,
        "minLength": 1,
        "type": "string",
        "description": "A tag that consists of letters, numbers, spaces, and special characters (`.`, `-`, `_`, `/`, `:`).\n",
        "example": "project-7"
      },
      "Tags": {
        "maxItems": 30,
        "type": "array",
        "description": "A list of tags that categorize the order. A tag can consist of letters, numbers, spaces, and special characters (`.`, `-`, `_`, `/`, `:`).\n",
        "items": {
          "$ref": "#/components/schemas/Tag"
        }
      },
      "OrderDisplayName": {
        "maxLength": 200,
        "minLength": 1,
        "type": "string",
        "description": "A human-readable name that describes the order.",
        "example": "Pléiades Neo over North America"
      },
      "CreateV2OrderRequest": {
        "required": [
          "dataProduct",
          "displayName",
          "featureCollection",
          "params"
        ],
        "type": "object",
        "properties": {
          "dataProduct": {
            "type": "string",
            "description": "The data product ID.",
            "format": "uuid",
            "example": "07c33a51-94b9-4509-84df-e9c13ea92b84"
          },
          "displayName": {
            "$ref": "#/components/schemas/OrderDisplayName"
          },
          "tags": {
            "$ref": "#/components/schemas/Tags"
          },
          "budgetId": {
            "type": "string",
            "description": "The ID of the budget selected for this order. Credit consumption is recorded when the order reaches a [consumption event](https://docs.up42.com/developers/api-budgets#consumption-events) associated with a specific status.\n",
            "format": "uuid",
            "nullable": true,
            "example": "3fa85f64-5717-4562-b3fc-2c963f66afa6"
          },
          "params": {
            "type": "object",
            "additionalProperties": true,
            "description": "Order parameters, excluding spatial geometry handled in the `featureCollection` object.\n",
            "example": {
              "spectralBands": "bundle",
              "radiometricProcessing": "reflectance",
              "geometricProcessing": "orthorectified",
              "pixelCoding": "16bit",
              "priority": "high",
              "maxIncidenceAngle": 20,
              "maxCloudCover": 20,
              "projection": "4326",
              "acquisitionMode": "mono",
              "acquisitionStart": "2025-11-20T00:00:00.000Z",
              "acquisitionEnd": "2025-11-27T00:00:00.000Z"
            }
          },
          "featureCollection": {
            "$ref": "#/components/schemas/OrderingFeatureCollection"
          }
        }
      },
      "EstimateV2OrderResponse": {
        "required": [
          "errors",
          "results",
          "summary"
        ],
        "type": "object",
        "properties": {
          "summary": {
            "$ref": "#/components/schemas/EstimationV2Summary"
          },
          "results": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/EstimationV2Result"
            }
          },
          "errors": {
            "type": "array",
            "description": "Errors associated with individual geometries, including validation failures due to AOI size or vertex-count limits.\n",
            "example": [],
            "items": {
              "$ref": "#/components/schemas/BatchEndpointErrorDetail"
            }
          }
        },
        "description": "The cost estimation.\n"
      },
      "EstimationV2Summary": {
        "required": [
          "totalCredits",
          "totalSize",
          "unit"
        ],
        "type": "object",
        "properties": {
          "totalCredits": {
            "type": "number",
            "description": "The estimate of the order cost, in credits.\n",
            "format": "int32",
            "nullable": false,
            "example": 14424,
            "default": 0
          },
          "totalSize": {
            "type": "number",
            "description": "The size of the order in square kilometers or the number of scenes needed to cover the requested geometry.",
            "nullable": false,
            "example": 8.48,
            "default": 0
          },
          "unit": {
            "$ref": "#/components/schemas/EstimateV2OrderPricingUnit"
          }
        },
        "description": "The overview of the estimation.\n"
      },
      "EstimationV2Result": {
        "required": [
          "credits",
          "index",
          "size",
          "unit"
        ],
        "type": "object",
        "properties": {
          "index": {
            "type": "number",
            "description": "The geometry index number. Indexing starts from zero.\n",
            "format": "int32"
          },
          "credits": {
            "type": "number",
            "description": "The estimated cost of the geometry, in credits.\n",
            "format": "int32",
            "example": 14424
          },
          "size": {
            "type": "number",
            "description": "The size of the geometry.",
            "example": 8.48
          },
          "unit": {
            "$ref": "#/components/schemas/EstimateV2OrderPricingUnit"
          }
        },
        "description": "Results for each of the specified geometries.\n"
      },
      "EstimateV2OrderPricingUnit": {
        "type": "string",
        "description": "The unit of measurement used to calculate the size.\n",
        "nullable": true,
        "enum": [
          "SQ_KM",
          "SCENE",
          null
        ]
      },
      "OrderingFeature": {
        "required": [
          "type"
        ],
        "type": "object",
        "properties": {
          "type": {
            "type": "string",
            "description": "The type of geometry. The only allowed value is `Feature`.\n",
            "enum": [
              "Feature"
            ]
          },
          "geometry": {
            "$ref": "#/components/schemas/OrderingGeometry"
          },
          "properties": {
            "type": "object"
          }
        },
        "description": "A spatially bounded GeoJSON feature.\n",
        "externalDocs": {
          "url": "https://tools.ietf.org/html/rfc7946#section-3.2"
        }
      },
      "OrderingFeatureCollection": {
        "required": [
          "features",
          "type"
        ],
        "type": "object",
        "properties": {
          "type": {
            "type": "string",
            "description": "The type of GeoJSON object. The only allowed value is `FeatureCollection`.\n",
            "enum": [
              "FeatureCollection"
            ]
          },
          "features": {
            "type": "array",
            "description": "GeoJSON features.\n",
            "items": {
              "$ref": "#/components/schemas/OrderingFeature"
            }
          }
        },
        "description": "A GeoJSON feature collection.\n",
        "externalDocs": {
          "url": "https://tools.ietf.org/html/rfc7946#section-3.3"
        }
      },
      "BatchEndpointErrorDetail": {
        "required": [
          "details",
          "index",
          "message"
        ],
        "type": "object",
        "properties": {
          "index": {
            "type": "number",
            "description": "The failed geometry index number.\n"
          },
          "message": {
            "type": "string",
            "description": "The error message associated with the failed geometry."
          },
          "details": {
            "type": "string",
            "description": "Error message details."
          }
        },
        "description": "Errors associated with individual geometries returned by batch endpoints, including validation failures due to AOI size or vertex-count limits.\n"
      },
      "OrderingGeometry": {
        "required": [
          "coordinates",
          "type"
        ],
        "type": "object",
        "properties": {
          "type": {
            "type": "string",
            "description": "The type of geometry.",
            "example": "Polygon",
            "enum": [
              "Polygon",
              "Point"
            ]
          },
          "coordinates": {
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
            "oneOf": [
              {
                "type": "array",
                "items": {
                  "type": "array",
                  "items": {
                    "minItems": 2,
                    "type": "array",
                    "items": {
                      "type": "number",
                      "format": "double"
                    }
                  }
                }
              },
              {
                "type": "array",
                "items": {
                  "type": "number",
                  "format": "double"
                }
              }
            ]
          },
          "crs": {
            "type": "object",
            "properties": {
              "type": {
                "type": "string",
                "description": "The parameter defining the CRS.",
                "example": "name"
              },
              "properties": {
                "type": "object",
                "properties": {
                  "name": {
                    "type": "string",
                    "description": "The name of the CRS.",
                    "example": "EPSG:4326"
                  }
                }
              }
            },
            "description": "The CRS of the order."
          }
        },
        "description": "The geometry in the GeoJSON format.\n",
        "externalDocs": {
          "url": "http://geojson.org/geojson-spec.html#geometry-objects"
        }
      },
      "InvalidParam": {
        "required": [
          "name",
          "reason"
        ],
        "type": "object",
        "properties": {
          "name": {
            "type": "string",
            "description": "The parameter or field name"
          },
          "reason": {
            "type": "string",
            "description": "A human-readable explanation of the validation failure"
          }
        }
      },
      "ErrorResponseV1": {
        "required": [
          "error"
        ],
        "type": "object",
        "properties": {
          "data": {
            "type": "object",
            "nullable": true
          },
          "error": {
            "$ref": "#/components/schemas/ErrorResponseObject"
          }
        },
        "description": "Legacy error wrapper used for authentication failures and some data-adapter schema errors."
      },
      "ErrorResponseObject": {
        "required": [
          "code",
          "message"
        ],
        "type": "object",
        "properties": {
          "code": {
            "type": "integer",
            "format": "int32"
          },
          "message": {
            "type": "string"
          },
          "details": {
            "type": "object",
            "nullable": true
          }
        }
      },
      "OrderingProblem": {
        "required": [
          "status",
          "title",
          "type"
        ],
        "type": "object",
        "properties": {
          "type": {
            "type": "string",
            "description": "A URI reference that uniquely identifies the problem type.\n",
            "format": "uri-reference"
          },
          "status": {
            "type": "integer",
            "description": "The HTTP status code generated by the origin server for this occurrence of the problem.\n"
          },
          "title": {
            "type": "string",
            "description": "A short summary of the problem type.\n"
          },
          "detail": {
            "type": "string",
            "description": "A human-readable explanation specific to this occurrence of the problem.\n",
            "nullable": true
          },
          "instance": {
            "type": "string",
            "description": "A URI reference that identifies the specific occurrence of the problem.\n",
            "format": "uri-reference",
            "nullable": true
          },
          "errors": {
            "type": "array",
            "description": "Validation errors for specific parameters.",
            "nullable": true,
            "items": {
              "$ref": "#/components/schemas/InvalidParam"
            }
          }
        },
        "description": "RFC 9457 Problem Details for HTTP APIs"
      }
    },
    "responses": {
      "OrderingUnauthorized": {
        "description": "Unauthorized",
        "content": {
          "application/json": {
            "schema": {
              "$ref": "#/components/schemas/ErrorResponseV1"
            },
            "example": {
              "data": null,
              "error": {
                "code": 401,
                "message": "An error occurred while attempting to decode the Jwt: Jwt expired at 2023-02-15T11:29:44Z",
                "details": null
              }
            }
          }
        }
      },
      "BadRequest": {
        "description": "Order request not valid",
        "content": {
          "application/problem+json": {
            "schema": {
              "$ref": "#/components/schemas/OrderingProblem"
            },
            "example": {
              "type": "https://docs.up42.com/problems/bad-order-request",
              "status": 400,
              "title": "Order request not valid",
              "detail": "One or more query parameter values are not valid.",
              "instance": "/v2/orders",
              "errors": [
                {
                  "name": "featureCollection",
                  "reason": "The feature list must not be empty."
                }
              ]
            }
          }
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