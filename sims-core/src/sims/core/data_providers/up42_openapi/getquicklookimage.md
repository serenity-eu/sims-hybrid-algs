---
updatedAt: 2026-05-27T08:26:46.000Z
---

Fetch the complete documentation index at: https://developer.up42.com/llms.txt. Use this file to discover all available pages before exploring further.

# Get a quicklook

Get a quicklook of a full scene from the catalog. Not all data hosts provide quicklooks.

Quicklooks are low-resolution previews of full scenes on a basemap.


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
    "/catalog/{host-name}/image/{image-id}/quicklook": {
      "get": {
        "tags": [
          "Catalog"
        ],
        "summary": "Get a quicklook",
        "description": "Get a quicklook of a full scene from the catalog. Not all data hosts provide quicklooks.\n\nQuicklooks are low-resolution previews of full scenes on a basemap.\n",
        "operationId": "getQuicklookImage",
        "parameters": [
          {
            "$ref": "#/components/parameters/HostNamePath"
          },
          {
            "$ref": "#/components/parameters/ImageIDPath"
          }
        ],
        "responses": {
          "200": {
            "description": "OK",
            "content": {
              "image/png": {
                "schema": {
                  "$ref": "#/components/schemas/PreviewResource"
                }
              },
              "image/jpeg": {
                "schema": {
                  "$ref": "#/components/schemas/PreviewResource"
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
          "542": {
            "$ref": "#/components/responses/HostError"
          }
        }
      }
    }
  },
  "components": {
    "schemas": {
      "PreviewResource": {
        "type": "string",
        "format": "byte"
      }
    },
    "responses": {
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
      },
      "ImageIDPath": {
        "name": "image-id",
        "in": "path",
        "description": "The full scene ID.",
        "required": true,
        "style": "simple",
        "explode": false,
        "schema": {
          "type": "string",
          "example": "TRIPLESAT_3_PMS_20230208011027_0042C9VI_002"
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