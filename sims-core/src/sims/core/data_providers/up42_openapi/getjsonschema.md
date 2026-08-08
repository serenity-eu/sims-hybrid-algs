---
updatedAt: 2026-05-27T08:26:46.000Z
---

Fetch the complete documentation index at: https://developer.up42.com/llms.txt. Use this file to discover all available pages before exploring further.

# Get a JSON schema of an order form

Get detailed information about the parameters needed to create an order for a specific data product.

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
    "/orders/schema/{data-product-id}": {
      "get": {
        "tags": [
          "Orders"
        ],
        "summary": "Get a JSON schema of an order form",
        "description": "Get detailed information about the parameters needed to create an order for a specific data product.",
        "operationId": "getJSONSchema",
        "parameters": [
          {
            "name": "data-product-id",
            "in": "path",
            "description": "The data product ID.",
            "required": true,
            "style": "simple",
            "explode": false,
            "schema": {
              "title": "Data Product Id",
              "type": "string",
              "description": "The data product ID.",
              "format": "uuid"
            },
            "example": "68567134-27ad-7bd7-4b65-d61adb11fc78"
          }
        ],
        "responses": {
          "200": {
            "description": "OK",
            "content": {
              "application/schema+json": {
                "schema": {
                  "type": "string"
                },
                "examples": {
                  "Tasking": {
                    "value": {
                      "$schema": "https://json-schema.org/draft-07/schema",
                      "type": "object",
                      "properties": {
                        "geometry": {
                          "description": "Polygon Model.",
                          "properties": {
                            "type": {
                              "const": "Polygon",
                              "default": "Polygon",
                              "title": "Type",
                              "type": "string"
                            },
                            "coordinates": {
                              "items": {
                                "items": {
                                  "items": {
                                    "type": "number"
                                  },
                                  "maxItems": 2,
                                  "minItems": 2,
                                  "type": "array"
                                },
                                "minItems": 4,
                                "type": "array"
                              },
                              "type": "array"
                            }
                          },
                          "required": [
                            "coordinates"
                          ],
                          "title": "Geometry",
                          "type": "object"
                        },
                        "displayName": {
                          "title": "Order name",
                          "type": "string"
                        },
                        "extraDescription": {
                          "title": "Description",
                          "type": "string"
                        },
                        "acquisitionStart": {
                          "format": "date-time",
                          "title": "Start",
                          "type": "string"
                        },
                        "acquisitionEnd": {
                          "format": "date-time",
                          "title": "End",
                          "type": "string"
                        },
                        "acquisitionMode": {
                          "anyOf": [
                            {
                              "const": "spot",
                              "title": "Spot (standard)"
                            },
                            {
                              "const": "spot_enhanced",
                              "title": "Spot (enhanced)"
                            },
                            {
                              "const": "spot_ultra",
                              "title": "Spot (ultra)"
                            },
                            {
                              "const": "strip",
                              "title": "Strip (standard)"
                            },
                            {
                              "const": "strip_enhanced",
                              "title": "Strip (enhanced)"
                            },
                            {
                              "const": "scan",
                              "title": "Scan (standard)"
                            },
                            {
                              "const": "scan_enhanced",
                              "title": "Scan (enhanced)"
                            }
                          ],
                          "title": "Acquisition mode",
                          "default": "spot",
                          "description": "The operation mode that determines azimuth resolution and swath width.",
                          "type": "string"
                        },
                        "maxIncidenceAngle": {
                          "default": 25,
                          "description": "The maximum allowed angle between the ground normal and look direction from the satellite.",
                          "maximum": 90,
                          "minimum": 0,
                          "title": "Maximum incidence angle (°)",
                          "type": "integer"
                        },
                        "polarization": {
                          "anyOf": [
                            {
                              "const": "hh",
                              "title": "HH"
                            },
                            {
                              "const": "vv",
                              "title": "VV"
                            },
                            {
                              "const": "vh",
                              "title": "VH"
                            },
                            {
                              "const": "hv",
                              "title": "HV"
                            }
                          ],
                          "title": "Polarization",
                          "default": "vv",
                          "description": "The direction of travel of an electromagnetic wave: vertical (V) or horizontal (H). The first letter corresponds to how signals are emitted, and the second letter corresponds to how they are received.",
                          "type": "string"
                        },
                        "looks": {
                          "anyOf": [
                            {
                              "const": "1",
                              "title": "1"
                            }
                          ],
                          "title": "Number of looks",
                          "default": "1",
                          "description": "The number of times a sensor captures the target. Single-look imagery will have more detail but also more noise. Multi-look imagery will be easier to understand but less detailed.",
                          "type": "string"
                        },
                        "priority": {
                          "anyOf": [
                            {
                              "const": "standard",
                              "title": "Standard"
                            },
                            {
                              "const": "high",
                              "title": "High"
                            }
                          ],
                          "title": "Priority",
                          "default": "standard",
                          "description": "The urgency of the order. High-priority orders are completed faster, but cost more.",
                          "type": "string"
                        }
                      },
                      "required": [
                        "geometry",
                        "displayName",
                        "acquisitionStart",
                        "acquisitionEnd",
                        "acquisitionMode",
                        "maxIncidenceAngle",
                        "polarization",
                        "priority",
                        "looks"
                      ],
                      "additionalProperties": false
                    }
                  },
                  "Catalog": {
                    "value": {
                      "additionalProperties": false,
                      "properties": {
                        "id": {
                          "title": "Id",
                          "type": "string"
                        },
                        "aoi": {
                          "description": "Polygon Model.",
                          "properties": {
                            "type": {
                              "const": "Polygon",
                              "default": "Polygon",
                              "title": "Type",
                              "type": "string"
                            },
                            "coordinates": {
                              "items": {
                                "items": {
                                  "items": {
                                    "type": "number"
                                  },
                                  "maxItems": 2,
                                  "minItems": 2,
                                  "type": "array"
                                },
                                "minItems": 4,
                                "type": "array"
                              },
                              "type": "array"
                            }
                          },
                          "required": [
                            "coordinates"
                          ],
                          "title": "Polygon",
                          "type": "object"
                        }
                      },
                      "required": [
                        "id",
                        "aoi"
                      ],
                      "title": "ParamsWithAOI",
                      "type": "object",
                      "$schema": "https://json-schema.org/draft-07/schema"
                    }
                  }
                }
              }
            }
          },
          "400": {
            "description": "Not valid data product ID",
            "content": {
              "application/problem+json": {
                "schema": {
                  "$ref": "#/components/schemas/Problem"
                },
                "example": {
                  "type": "https://docs.up42.com/problems/not-valid-uuid",
                  "title": "Not a valid UUID",
                  "status": 400
                }
              }
            }
          },
          "404": {
            "description": "Resource not found",
            "content": {
              "application/problem+json": {
                "schema": {
                  "$ref": "#/components/schemas/Problem"
                },
                "examples": {
                  "Schema not found": {
                    "value": {
                      "type": "https://docs.up42.com/problems/schema-not-found",
                      "title": "Schema not found",
                      "status": 404
                    }
                  },
                  "Data product not found": {
                    "value": {
                      "type": "https://docs.up42.com/problems/data-product-not-found",
                      "title": "Data product not found",
                      "status": 404
                    }
                  }
                }
              }
            }
          }
        },
        "security": [
          {
            "httpBearer": []
          }
        ]
      }
    }
  },
  "components": {
    "schemas": {
      "Problem": {
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
            "description": "A human-readable explanation specific to this occurrence of the problem\n",
            "nullable": true
          }
        },
        "description": "RFC 9457 problem details for HTTP APIs"
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