//
//  Item.swift
//  Todoey_V_3.0
//
//  Created by Eric Magliarditi on 6/24/19.
//  Copyright © 2019 Eric Magliarditi. All rights reserved.
//

import Foundation
import RealmSwift

class Item: Object {
    @objc dynamic var title: String = ""
    @objc dynamic var done: Bool = false
    @objc dynamic var dateCreated: Date?
    
    /**
     Need to create the parent child relationship - already astablished in the category but need to create the inverse relationship - this is not automatically generated as in CoreData
     Use Realm Object = LinkingObjects
        these are objects that define the inverse relationship to a category
     need Category.self to ensure you get the Type not the class itself
     The string is the same name as the items we generated in the category.swift file
    */
    var parentCategory = LinkingObjects(fromType: Category.self, property: "items")
}
